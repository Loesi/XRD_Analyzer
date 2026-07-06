

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pydantic import BaseModel, model_validator
from typing import Any, Optional, Union, List, Tuple, Literal, cast
from ipywidgets import IntSlider, FloatLogSlider, HBox
from IPython.display import display
from matplotlib import axes, lines
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

import h5py, lmfit, pathlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pybaselines as pb
import scipy.constants as sc

numeric = Union[int, float]
num_Non = Union[numeric, None]


ProcessedState = Literal["raw", "corrected", "normalized"]
#region XRD_Data
@dataclass
class XRD_Data:
    q: npt.NDArray[np.floating[Any]]
    time: npt.NDArray[np.floating[Any]]
    intensity: npt.NDArray[np.floating[Any]]
    energy: numeric

    _baseline_lam: float = 0
    _norm_slice: Optional[slice] = None

    _processed: npt.NDArray[np.floating[Any]] = field(init=False, repr=False)
    _processed_state: ProcessedState = "raw"
    _reflexes: Optional[tuple[dict[str, Any], List[npt.NDArray[np.floating[Any]]]]] = None
    

    @classmethod
    def extract_data_hdf5(cls, file: pathlib.Path | str, energy: numeric) -> 'XRD_Data':
        print(file)
        with h5py.File(file, 'r') as run_datafile:
            entry = cast(h5py.Group, run_datafile["entry"])
            data1d = cast(h5py.Group, entry["data1d"])

            intensity = np.array(cast(h5py.Dataset, data1d["I"]))
            if "q" in data1d:
                q = np.array(cast(h5py.Dataset, data1d["q"]))
            elif "2th" in data1d:
                q = np.sin(np.clip(np.deg2rad(np.array(cast(h5py.Dataset, data1d["2th"])))/2, -1, 1)) * (energy * sc.e *  4 * sc.pi) / (1E10 * sc.c * sc.h)
            else:
                raise ValueError(f"did not find angle data in {file}, expected 'entry/data1d/q' or 'entry/data1d/2th' dataset but found neither.")

        return cls(q, np.arange(intensity.shape[0]), intensity, energy)

    def __post_init__(self):
        self.removeNan()
        self.validate()
        self._processed = np.empty(shape = self.intensity.shape, dtype=np.float64)

    def removeNan(self):
        has_nan = np.isnan(self.intensity).any(axis=1)
        self.intensity = self.intensity[~has_nan]
        self.time = self.time[~has_nan]
        
    def validate(self):
        """Validate shapes on creation."""
        self._validate_array(self.time, "time", expected_dim=1)
        self._validate_array(self.q, "energies", expected_dim=1)
        self._validate_array(self.intensity, "intensity", expected_dim=2)

        M, N = self.intensity.shape
        if (len(self.time) != M) or (len(self.q) != N):
            raise ValueError(f"Expected intensity data of shape {len(self.time)}, {len(self.q)}, recieved {self.intensity.shape}")
        
    def _validate_array(self, arr: npt.NDArray, name: str, expected_dim: int):
        """Helper to check if an array is valid, of right dimension and numeric."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Field '{name}' must be a numpy ndarray, got {type(arr)}")
        
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Field '{name}' must contain numeric data, got {arr.dtype}")

        if arr.ndim != expected_dim:
            raise ValueError(f"Field '{name}' must be {expected_dim}D, but got {arr.ndim}D with shape {arr.shape}")
        
        if arr.size == 0:
            raise ValueError(f"Field '{name}' cannot be empty.")
        
    @property
    def twoTheta(self):
        return np.rad2deg(2 * np.arcsin(self.q * 1E10 * sc.c * sc.h / (self.energy * sc.e *  4 * sc.pi)))
    
    @property
    def baseline_lam(self) -> float:
        return self._baseline_lam

    @baseline_lam.setter
    def baseline_lam(self, value: float) -> None:
        if value != self._baseline_lam:
            self._baseline_lam = value
            self._processed_state = "raw"

    @property
    def norm_slice(self) -> slice:
        assert (self._norm_slice is not None)
        return self._norm_slice

    @norm_slice.setter
    def norm_slice(self, value: slice) -> None:
        if value != self._norm_slice:
            self._norm_slice = value
            if self._processed_state == "normalized":
                self._processed_state = "corrected"
    
    @property
    def processed(self):
        if (self._processed_state == "raw") and self._baseline_lam == 0:
            raise BaseException("To use processed data at least a baseline correction has to be applied.")
        
        if self._processed_state == "raw":
            baseline = pb.Baseline(x_data=self.q)
        
            for i in range(len(self.intensity)):
                self._processed[i] = self.intensity[i] - baseline.arpls(self.intensity[i], lam=self._baseline_lam)[0]
            self._processed_state = "corrected"
            print("corrected")

        if (self._processed_state == "corrected") and (self._norm_slice is not None):
            integrals = np.trapezoid(y = self._processed[:, self._norm_slice], x = self.q[self._norm_slice], axis=1)

            assert isinstance(integrals, np.ndarray)
            self._processed /= integrals[:, None]
            self._processed_state = "normalized"
            print("normalized")

        return self._processed
    
    def save_corrected_pd(self, file):

        df = pd.DataFrame(self.processed, index=self.time, columns=self.q).T
        df.to_csv(file)

    def _fit_reflexes(self, i: int, snr: numeric, min_length:int, height_mlt: numeric, distance: numeric,
                   delta_pos: numeric, default_sigma: numeric) -> npt.NDArray[np.floating[Any]]:

        data = np.copy(self.processed[i])
        diff = np.diff((data > snr * np.median(data)).astype(int))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

        if starts[0] > ends[0]:
            ends = ends [1:]
        if starts[-1] > ends[-1]:
            starts = starts[:-1]
        assert len(starts) == len(ends), f"unmatches sizes {starts.shape} - {ends.shape}"
        areas = [slice(s,e) for s,e in zip(starts,ends) if e-s >= min_length]

        dq = np.average(np.diff(self.q))
        two_theta = self.twoTheta
        data = gaussian_filter(data, 5)
        reflexes = []
        for s in areas:
            grad = np.gradient(np.gradient(data[s], dq), dq)
            rs = find_peaks(-grad, height_mlt * np.abs(grad).max(), distance = distance)[0]
            for r in rs:
                reflexes.append(s.start + r)
        
        model = None
        params = lmfit.Parameters()
        for j,l in enumerate(reflexes):
            prefix = f"r{j}_"
            cur_refl = lmfit.models.PseudoVoigtModel(prefix=prefix)
            if model is None:
                model = cur_refl
            else:
                model += cur_refl

            pos = two_theta[l]
            amp = data[l]
            params.update(cur_refl.make_params())
            params[prefix + "center"].set(value=pos, min=pos-delta_pos, max=pos+delta_pos)
            params[prefix + "amplitude"].set(value=amp * default_sigma * np.pi)
            params[prefix + "sigma"].set(value=default_sigma*1)
            params[prefix + "fraction"].set(value=.2, min=0, max=1, vary=True)
        
        assert isinstance(model, lmfit.Model)
        result = model.fit(data, params, x=two_theta)
        # result = model.fit(data, params, x=two_theta, method="lbfgsb")

        prefs = [m.prefix for m in model.components]

        print(f"{i} finished")
        return np.array([[
            result.params[p + "center"].value,
            result.params[p + "amplitude"].value,
            result.params[p + "sigma"].value,
            result.params[p + "fraction"].value
        ] for p in prefs])

    def reflexes(self,
              energy: numeric,
              snr: numeric = 20,
              min_length:int = 10,
              height_mlt: numeric = .5,
              distance: numeric = 20,
              delta_pos: numeric = 0.025,
              default_sigma: numeric = 0.05
        ) -> List[npt.NDArray[np.floating[Any]]]:
        
        args_dict = locals() 
        if self._reflexes is not None and self._reflexes[0] == args_dict:
            return self._reflexes[1]

        print("recalculating reflexes")
        worker = partial(self._fit_reflexes, snr=snr, min_length=min_length, height_mlt=height_mlt,
                            distance=distance, delta_pos= delta_pos, default_sigma=default_sigma)

        iterator = range(len(self.time))
        # iterator = range(1)
        with ProcessPoolExecutor() as ex:
            res = ex.map(worker, iterator)
        reflexes = list(res)

        self._reflexes = (args_dict, reflexes)
        return reflexes

#region REFLEX & REF
@dataclass
class REFLEX:
    q: float
    intensity: float
    width: Optional[float]

@dataclass    
class XRD_REF:
    reflexes: Any
    diff: Optional[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]] = None

#region XRD_Analyzer
class XRD_Analyzer(BaseModel):
    model_config={"arbitrary_types_allowed": True}
    data: XRD_Data
    # refs: list[XRD_REF]

    @model_validator(mode='before')
    @classmethod
    def from_file(cls, conf: Any) -> Any:
        if not isinstance(conf, dict):
            return conf
        
        if "data" not in conf.keys():
            return conf
        
        if not isinstance(conf["data"], str | pathlib.Path):
            raise ValueError('XRD_Analyzer can only be called with an XRD_Data object or a filepath toward the data.')
        
        conf["data"] = XRD_Data.extract_data_hdf5(conf["data"], conf["energy"])
        return conf
        
    def baselineCorrection(self):
        baseline = pb.Baseline(x_data=self.data.q)

        lamSlider: FloatLogSlider = FloatLogSlider(value=1E4, base=10, min=3, max=6, step=0.01, description="lam")
        tSlider: IntSlider = IntSlider(value=0, min=0, max=len(self.data.time)-1, description="#Diffractogram")


        fig, (ax1, ax2) = plt.subplots(2,1, layout="tight", figsize= (8,6))
        y_data = self.data.intensity[tSlider.value]
        y_corr, params = baseline.arpls(y_data, lam=lamSlider.value)
        l_data = ax1.plot(self.data.q, y_data)[0]
        l_corr = ax1.plot(self.data.q, y_corr)[0]
        l_norm = ax2.plot(self.data.q, y_data-y_corr)[0]

        def update(val: Any):
            y_data = self.data.intensity[tSlider.value]
            y_corr, params = baseline.arpls(y_data, lam=lamSlider.value)
            l_data.set_ydata(y_data)
            l_corr.set_ydata(y_corr)
            l_norm.set_ydata(y_data-y_corr)
            self.data.baseline_lam = lamSlider.value

        lamSlider.observe(update)
        tSlider.observe(update)

        display(HBox([tSlider,lamSlider]))

    def normalizeByRegion(self, lower: numeric, upper:numeric):
        lower_idx, upper_idx = np.searchsorted(self.data.q, lower), np.searchsorted(self.data.q, upper)
        self.data.norm_slice = slice(lower_idx, upper_idx)

    def export_processed(self, exp_path: pathlib.Path, in_q = False):

        with exp_path.open("w") as f:
            f.write(",".join(["Two_theta"] + [str(a) for a in self.data.time]) + "\n")
            x = self.data.q if in_q else self.data.twoTheta
            for i, v in enumerate(x):
                f.write(",".join([str(v)] + [str(a) for a in self.data.processed[:, i]]) + "\n")

        print(f"successfully written to {exp_path}")


    def plot2D(self, in_q: bool = False, **kwargs):
        plt.close()
        corrected_data = self.data.processed
        x_data = self.data.q if in_q else self.data.twoTheta

        fig, ax = plt.subplots(**kwargs)
        assert isinstance(ax, axes.Axes)
        xs, ys = np.meshgrid(x_data, np.arange(len(self.data.time)))
        ax.pcolormesh(xs, ys, corrected_data, shading="auto")
        # ax.imshow(corrected_data, aspect="auto",extent=(self.data.q[0], self.data.q[-1], 0, len(self.data.intensity)), origin="lower")
        ax.set_xlabel("q [A$^{-1}$]") if in_q else ax.set_xlabel(r"2$\Theta$ [°]")
        ax.set_ylabel("Scan #")
        return fig, ax
    
    def plot1D(self, i: int, in_q: bool = False, x_min: num_Non = None, x_max: num_Non = None, **kwargs):
        plt.close()
        corrected_data = self.data.processed
        x_data = self.data.q if in_q else self.data.twoTheta
        x_min = x_data[0] if x_min is None else x_min
        x_max = x_data[-1] if x_max is None else x_max

        fig, ax = plt.subplots(**kwargs)
        ax.set_xlim(x_min,x_max)
        ax.plot(x_data, corrected_data[i])

        ax.set_xlabel("q [A$^{-1}$]") if in_q else ax.set_xlabel(r"2$\Theta$ [°]")
        ax.set_ylabel("intensity [A.U.]")

        return fig, ax






def XRD_File(file: str|pathlib.Path, energy: numeric) -> XRD_Analyzer:
    return XRD_Analyzer.model_validate({"data": file, "energy": energy})