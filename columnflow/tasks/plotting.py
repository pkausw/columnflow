# coding: utf-8

"""
Tasks to plot different types of histograms.
"""

from collections import OrderedDict
from abc import abstractmethod

import law
import luigi

from columnflow.tasks.framework.base import ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin,
    CategoriesMixin, ShiftSourcesMixin, EventWeightMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, PlotBase2D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
)
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms, MergeShiftedHistograms
from columnflow.util import DotDict, dev_sandbox, dict_add_strict


class PlotVariablesBase(
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    EventWeightMixin,
    CategoriesMixin,
    PlotBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    exclude_index = True

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "plot", f"datasets_{self.datasets_repr}")
        return parts

    def workflow_requires(self, only_super: bool = False):
        reqs = super().workflow_requires()
        if only_super:
            return reqs

        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    @abstractmethod
    def get_plot_shifts(self):
        return

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process
        hists = {}

        with self.publish_step(f"plotting {self.branch_data.variable} in {category_inst.name}"):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)
                h_in = inp["collection"][0].targets[self.branch_data.variable].load(formatter="pickle")

                # loop and extract one histogram per process
                for process_inst in process_insts:
                    # skip when the dataset is already known to not contain any sub process
                    if not any(map(dataset_inst.has_process, sub_process_insts[process_inst])):
                        continue

                    # work on a copy
                    h = h_in.copy()

                    # axis selections
                    h = h[{
                        "process": [
                            hist.loc(p.id)
                            for p in sub_process_insts[process_inst]
                            if p.id in h.axes["process"]
                        ],
                        "category": [
                            hist.loc(c.id)
                            for c in leaf_category_insts
                            if c.id in h.axes["category"]
                        ],
                        "shift": [
                            hist.loc(s.id)
                            for s in plot_shifts
                            if s.id in h.axes["shift"]
                        ],
                    }]

                    # axis reductions
                    h = h[{"process": sum, "category": sum}]

                    # add the histogram
                    if process_inst in hists:
                        hists[process_inst] += h
                    else:
                        hists[process_inst] = h

            # there should be hists to plot
            if not hists:
                raise Exception("no histograms found to plot")

            # sort hists by process order
            hists = OrderedDict(
                (process_inst, hists[process_inst])
                for process_inst in sorted(hists, key=process_insts.index)
            )

            # determine the correct plot function for this variable
            plot_function = (
                self.plot_function_1d if len(variable_insts) == 1 and "plot_function_1d" in dir(self) else
                self.plot_function_2d if len(variable_insts) == 2 and "plot_function_2d" in dir(self) else
                None
            )
            if not plot_function:
                raise NotImplementedError(
                    f"No Plotting function for {len(variable_insts)} variables implemented of task {self.task_family}",
                )

            # call the plot function
            fig = self.call_plot_func(
                plot_function,
                hists=hists,
                config_inst=self.config_inst,
                variable_insts=variable_insts,
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output():
                outp.dump(fig, formatter="mpl")


class PlotVariablesBaseSingleShift(
    ShiftTask,
    PlotVariablesBase,
):
    shifts = set(MergeHistograms.shifts)

    # default upstream dependency task classes
    dep_MergeHistograms = MergeHistograms

    exclude_index = True

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for var_name in self.variables
            for cat_name in self.categories
        ]

    def requires(self):
        return {
            d: self.dep_MergeHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def output(self):
        b = self.branch_data
        return [
            self.target(name)
            for name in self.get_plot_names(f"plot__proc_{self.processes_repr}__cat_{b.category}__var_{b.variable}")
        ]

    def get_plot_shifts(self):
        return [self.shift_inst]


class PlotVariables1D(
    PlotVariablesBaseSingleShift,
    PlotBase1D,
):
    pass


class PlotVariables2D(
    PlotVariablesBaseSingleShift,
    PlotBase2D,
):
    pass


class PlotVariablesPerProcess2D(
    law.WrapperTask,
    PlotVariables2D,
):
    # force this one to be a local workflow
    workflow = "local"

    def requires(self):
        return {
            process: PlotVariables2D.req(self, processes=(process,))
            for process in self.processes
        }


class PlotVariablesBaseMultiShifts(
    ShiftSourcesMixin,
    PlotVariablesBase,
):
    legend_title = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="sets the title of the legend; when empty and only one process is present in "
        "the plot, the process_inst label is used; empty default",
    )
    plot_function_1d = luigi.Parameter(
        default="columnflow.plotting.example.plot_shifted_variable",
        significant=False,
        description="name of the 1d plot function; default: 'columnflow.plotting.example.plot_shifted_variable'",
    )

    # default upstream dependency task classes
    dep_MergeShiftedHistograms = MergeShiftedHistograms

    exclude_index = True

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name, "shift_source": source})
            for var_name in self.variables
            for cat_name in self.categories
            for source in self.shift_sources
        ]

    def requires(self):
        return {
            d: self.dep_MergeShiftedHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def output(self):
        b = self.branch_data
        return [
            self.target(name)
            for name in self.get_plot_names(
                f"plot__proc_{self.processes_repr}__unc_{b.shift_source}__cat_{b.category}__var_{b.variable}",
            )
        ]

    def get_plot_shifts(self):
        return [
            self.config_inst.get_shift(s) for s in
            ["nominal", f"{self.branch_data.shift_source}_up", f"{self.branch_data.shift_source}_down"]
        ]

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()
        dict_add_strict(params, "legend_title", None if self.legend_title == law.NO_STR else self.legend_title)
        return params


class PlotShiftedVariables1D(
    PlotVariablesBaseMultiShifts,
    PlotBase1D,
):
    pass


class PlotShiftedVariablesPerProcess1D(
    law.WrapperTask,
    PlotShiftedVariables1D,
):
    # force this one to be a local workflow
    workflow = "local"

    def requires(self):
        return {
            process: PlotShiftedVariables1D.req(self, processes=(process,))
            for process in self.processes
        }
