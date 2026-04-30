from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from open_wam.configs.enums import StrEnum


class FlowUniPCPredictionType(StrEnum):
    FLOW_PREDICTION = "flow_prediction"


class FlowUniPCSolverType(StrEnum):
    BH1 = "bh1"
    BH2 = "bh2"


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """DreamZero-style UniPC sampler adapted for flow-matching prediction.

    This is a local copy of the flow-oriented UniPC scheduler used by DreamZero.
    We keep it under `src/` so register-attached runtime semantics are explicit
    and independent of the vendored previous-work tree.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: FlowUniPCPredictionType | str = FlowUniPCPredictionType.FLOW_PREDICTION,
        shift: float = 1.0,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: FlowUniPCSolverType | str = FlowUniPCSolverType.BH2,
        lower_order_final: bool = True,
        disable_corrector: List[int] | None = None,
    ) -> None:
        if solver_type not in {FlowUniPCSolverType.BH1, FlowUniPCSolverType.BH2}:
            raise NotImplementedError(f"{solver_type} is not implemented for flow UniPC.")
        self.predict_x0 = predict_x0
        self.num_inference_steps: int | None = None
        self.disable_corrector = disable_corrector or []
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.model_outputs: list[torch.Tensor | None] = [None] * solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1
        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        *,
        shift: float | None = None,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        shift_value = float(self.config.shift if shift is None else shift)
        sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]
        sigmas = shift_value * sigmas / (1 + (shift_value - 1) * sigmas)
        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value).unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample.to(dtype)

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if not self.predict_x0 or self.config.prediction_type != FlowUniPCPredictionType.FLOW_PREDICTION:
            raise ValueError("FlowUniPCMultistepScheduler only supports predict_x0 flow_prediction mode.")
        sigma_t = self.sigmas[step_index].to(device=sample.device, dtype=sample.dtype)
        x0_pred = sample - sigma_t * model_output
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        return x0_pred

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        assert m0 is not None
        x = sample

        sigma_t = self.sigmas[step_index + 1].to(device=sample.device, dtype=sample.dtype)
        sigma_s0 = self.sigmas[step_index].to(device=sample.device, dtype=sample.dtype)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device=sample.device, dtype=sample.dtype))
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=sample.dtype, device=sample.device))
        rks_tensor = torch.stack(rks, dim=0)
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        b = []
        r = []
        b_h = hh if self.config.solver_type == FlowUniPCSolverType.BH1 else torch.expm1(hh)
        for i in range(1, order + 1):
            r.append(torch.pow(rks_tensor, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        r_tensor = torch.stack(r, dim=0)
        b_tensor = torch.stack(b, dim=0)
        if d1s:
            d1_tensor = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.full((1,), 0.5, dtype=sample.dtype, device=sample.device)
            else:
                rhos_p = torch.linalg.solve_ex(
                    r_tensor[:-1, :-1].to(dtype=torch.float32),
                    b_tensor[:-1].to(dtype=torch.float32),
                )[0].to(sample.dtype)
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1_tensor)
        else:
            pred_res = 0
        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        x_t = x_t - alpha_t * b_h * pred_res
        return x_t.to(sample.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        assert m0 is not None
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t = self.sigmas[step_index].to(device=x.device, dtype=x.dtype)
        sigma_s0 = self.sigmas[step_index - 1].to(device=x.device, dtype=x.dtype)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device=x.device, dtype=x.dtype))
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=x.dtype, device=x.device))
        rks_tensor = torch.stack(rks, dim=0)
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        b = []
        r = []
        b_h = hh if self.config.solver_type == FlowUniPCSolverType.BH1 else torch.expm1(hh)
        for i in range(1, order + 1):
            r.append(torch.pow(rks_tensor, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        r_tensor = torch.stack(r, dim=0)
        b_tensor = torch.stack(b, dim=0)
        if d1s:
            d1_tensor = torch.stack(d1s, dim=1)
        else:
            d1_tensor = None
        if order == 1:
            rhos_c = torch.full((1,), 0.5, dtype=x.dtype, device=x.device)
        else:
            rhos_c = torch.linalg.solve_ex(
                r_tensor.to(dtype=torch.float32),
                b_tensor.to(dtype=torch.float32),
            )[0].to(x.dtype)

        x_t_base = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if d1_tensor is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1_tensor)
        else:
            corr_res = 0
        d1_t = model_t - m0
        x_t = x_t_base - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(x.dtype)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        step_index: int,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        if self.num_inference_steps is None:
            raise ValueError("Call `set_timesteps` before FlowUniPC sampling.")
        use_corrector = step_index > 0 and step_index - 1 not in self.disable_corrector and self.last_sample is not None
        model_output_convert = self.convert_model_output(
            model_output=model_output,
            sample=sample,
            step_index=step_index,
        )
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                step_index=step_index,
            ).clone()
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep
        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
            step_index=step_index,
        ).clone()
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return self.config.num_train_timesteps
