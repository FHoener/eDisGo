import os
import logging
import pandas as pd

from edisgo.network.topology import Topology
from edisgo.network.results import Results
from edisgo.network import timeseries
from edisgo.tools import pypsa_io, plots, tools, networkx_helper
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.io.ding0_import import import_ding0_grid
from edisgo.io.generators_import import oedb as import_generators_oedb, add_and_connect_mv_generator
from edisgo.tools.config import Config
from edisgo.tools.geo import find_nearest_bus
from edisgo.opf.run_mp_opf import run_mp_opf
from edisgo.opf.results.opf_result_class import OPFResults

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import Point

logger = logging.getLogger("edisgo")


class EDisGo:
    """
    Provides the top-level API for invocation of data import, power flow
    analysis, network reinforcement and flexibility measures.

    Parameters
    ----------
    worst_case_analysis : None or :obj:`str`, optional
        If not None time series for feed-in and load will be generated
        according to the chosen worst case analysis.
        Possible options are:

        * 'worst-case'

          Feed-in and load for the two worst-case scenarios feed-in case and
          load case are generated.

        * 'worst-case-feedin'

          Feed-in and load for the worst-case scenario feed-in case is
          generated.

        * 'worst-case-load'

          Feed-in and load for the worst-case scenario load case is generated.

        Be aware that if you choose to conduct a worst-case analysis your
        input for the following parameters will not be used:

        * `timeseries_generation_fluctuating`
        * `timeseries_generation_dispatchable`
        * `timeseries_load`

    ding0_grid : :obj:`str`
        Path to directory containing csv files of network to be loaded.
    config_path : None or :obj:`str` or :obj:`dict`
        Path to the config directory. Options are:

        * None

          If `config_path` is None, configs are loaded from the edisgo
          default config directory ($HOME$/.edisgo). If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.

        * :obj:`str`

          If `config_path` is a string, configs will be loaded from the
          directory specified by `config_path`. If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.

        * :obj:`dict`

          A dictionary can be used to specify different paths to the
          different config files. The dictionary must have the following
          keys:

          * 'config_db_tables'
          * 'config_grid'
          * 'config_grid_expansion'
          * 'config_timeseries'

          Values of the dictionary are paths to the corresponding
          config file. In contrast to the other two options, the directories
          and config files must exist and are not automatically created.

        Default: None.
    timeseries_generation_fluctuating : :obj:`str` or :pandas:`pandas.DataFrame<DataFrame>`
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:

        * 'oedb'

          Time series for the year 2011 are obtained from the OpenEnergy
          DataBase.

        * :pandas:`pandas.DataFrame<DataFrame>`

          DataFrame with time series, normalized with corresponding capacity.
          Time series can either be aggregated by technology type or by type
          and weather cell ID. In the first case columns of the DataFrame are
          'solar' and 'wind'; in the second case columns need to be a
          :pandas:`pandas.MultiIndex<MultiIndex>` with the first level
          containing the type and the second level the weather cell id.
          Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

         .. ToDo: explain how to obtain weather cell id,

         .. ToDo: add link to explanation of weather cell id

    timeseries_generation_dispatchable : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with time series for active power of each (aggregated)
        type of dispatchable generator normalized with corresponding capacity.
        Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Columns represent generator type:

        * 'gas'
        * 'coal'
        * 'biomass'
        * 'other'
        * ...

        Use 'other' if you don't want to explicitly provide every possible
        type.
    timeseries_generation_reactive_power : :pandas:`pandas.DataFrame<DataFrame>`, optional
        DataFrame with time series of normalized reactive power (normalized by
        the rated nominal active power) per technology and weather cell. Index
        needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Columns represent generator type and can be a MultiIndex column
        containing the weather cell ID in the second level. If the technology
        doesn't contain weather cell information i.e. if it is other than solar
        and wind generation, this second level can be left as a numpy Nan or a
        None.
        Default: None.
        If no time series for the technology or technology and weather cell ID
        is given, reactive power will be calculated from power factor and
        power factor mode in the config sections `reactive_power_factor` and
        `reactive_power_mode` and a warning will be raised.
    timeseries_load : :obj:`str` or :pandas:`pandas.DataFrame<DataFrame>`
        Parameter used to obtain time series of active power of loads.
        Possible options are:

        * 'demandlib'

          Time series for the year specified in `timeindex` are
          generated using the oemof demandlib.

        * :pandas:`pandas.DataFrame<DataFrame>`

          DataFrame with load time series of each type of load
          normalized with corresponding annual energy demand. Index needs to
          be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
          Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

    timeseries_load_reactive_power : :pandas:`pandas.DataFrame<DataFrame>`, optional
        DataFrame with time series of normalized reactive power (normalized by
        annual energy demand) per load sector. Index needs to be a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

        Default: None.
        If no time series for the load sector is given, reactive power will be
        calculated from power factor and power factor mode in the config
        sections `reactive_power_factor` and `reactive_power_mode` and a
        warning will be raised.
    generator_scenario : None or :obj:`str`
        If provided defines which scenario of future generator park to use
        and invokes import of these generators. Possible options are 'nep2035'
        and 'ego100'.

        .. ToDo: Add link to explanation of scenarios.

    timeindex : None or :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Can be used to select time ranges of the feed-in and load time series
        that will be used in the power flow analysis. Also defines the year
        load time series are obtained for when choosing the 'demandlib' option
        to generate load time series.

    Attributes
    ----------
    topology : :class:`~.network.topology.Topology`
        The topology is a container object holding the topology of the grids.
    timeseries: :class:`~.network.timeseries.TimeSeries`
        Container for component timeseries.
    results : :class:`~.network.results.Results`
        This is a container holding alls calculation results from power flow
        analyses, curtailment, storage integration, etc.

    """

    def __init__(self, **kwargs):

        # load configuration
        self._config = Config(config_path=kwargs.get("config_path", None))

        # instantiate topology object and load grid and equipment data
        self.topology = Topology(config=self.config)
        self.import_ding0_grid(path=kwargs.get("ding0_grid", None))

        # set up results and time series container
        self.results = Results(self)
        self.opf_results = OPFResults()
        self.timeseries = timeseries.TimeSeries()

        # import new generators
        if kwargs.get("generator_scenario", None) is not None:
            self.import_generators(kwargs.get("generator_scenario"))

        # set up time series for feed-in and load
        # worst-case time series
        if kwargs.get("worst_case_analysis", None):
            timeseries.get_component_timeseries(
                edisgo_obj=self, mode=kwargs.get("worst_case_analysis", None)
            )
        else:
            timeseries.get_component_timeseries(
                edisgo_obj=self,
                timeseries_generation_fluctuating=kwargs.get(
                    "timeseries_generation_fluctuating", None
                ),
                timeseries_generation_dispatchable=kwargs.get(
                    "timeseries_generation_dispatchable", None
                ),
                timeseries_generation_reactive_power=kwargs.get(
                    "timeseries_generation_reactive_power", None
                ),
                timeseries_load=kwargs.get("timeseries_load", None),
                timeseries_load_reactive_power=kwargs.get(
                    "timeseries_load_reactive_power", None
                ),
                timeseries_storage_units=kwargs.get(
                    "timeseries_storage_units", None
                ),
                timeseries_storage_units_reactive_power=kwargs.get(
                    "timeseries_storage_units_reactive_power", None
                ),
                timeindex=kwargs.get("timeindex", None),
            )

    @property
    def config(self):
        """
        eDisGo configuration data.

        Returns
        -------
        :class:`~.tools.config.Config`
            Config object with configuration data from config files.

        """
        return self._config

    @config.setter
    def config(self, config_path):
        self._config = Config(config_path=config_path)

    def import_ding0_grid(self, path):
        """
        Import ding0 topology data from csv files in the format as
        `Ding0 <https://github.com/openego/ding0>`_ provides it.

        Parameters
        -----------
        path : str
            Path to directory containing csv files of network to be loaded.

        """
        if path is not None:
            import_ding0_grid(path, self)

    def to_pypsa(self, **kwargs):
        """
        Convert to PyPSA network representation

        A network topology representation based on
        :pandas:`pandas.DataFrame<DataFrame>`. The overall container object of
        this data model, the :pypsa:`pypsa.Network<network>`,
        is set up.

        Parameters
        ----------
        kwargs :
            See :func:`~.tools.pypsa_io.to_pypsa` for further information.

        Returns
        -------
        :pypsa:`pypsa.Network<network>`
            PyPSA network representation.

        """
        timesteps = kwargs.get("timesteps", None)
        kwargs.pop("timesteps", None)
        mode = kwargs.get("mode", None)

        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]
        # export grid
        if not mode:
            return pypsa_io.to_pypsa(self, timesteps, **kwargs)
        elif "mv" in mode:
            return pypsa_io.to_pypsa(
                self.topology.mv_grid, timesteps, **kwargs
            )
        elif mode == "lv":
            lv_grid_name = kwargs.get("lv_grid_name", None)
            if not lv_grid_name:
                raise ValueError(
                    "For exporting lv grids, name of lv_grid has "
                    "to be provided."
                )
            return pypsa_io.to_pypsa(
                self.topology._grids[lv_grid_name],
                mode=mode,
                timesteps=timesteps
            )
        else:
            raise ValueError("The entered mode is not a valid option.")

    def to_graph(self):
        """
        Returns graph representation of the grid.

        Returns
        -------
        :networkx:`networkx.Graph<network.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

        """

        graph = networkx_helper.translate_df_to_graph(
            self.topology.buses_df,
            self.topology.lines_df,
            self.topology.transformers_df,
        )

        return graph

    def curtail(self, methodology, curtailment_timeseries, **kwargs):
        """
        Sets up curtailment time series.

        Curtailment time series are written into
        :class:`~.network.network.TimeSeries`. See
        :class:`~.network.network.CurtailmentControl` for more information on
        parameters and methodologies.

        """
        raise NotImplementedError
        # CurtailmentControl(edisgo=self, methodology=methodology,
        #                    curtailment_timeseries=curtailment_timeseries,
        #                    mode=kwargs.pop('mode', None), **kwargs)

    def import_generators(self, generator_scenario=None,
                          **kwargs):
        """Import generators

        For details see
        :func:`edisgo.io.generators_import.oedb`

        """
        if generator_scenario:
            self.topology.generator_scenario = generator_scenario
        import_generators_oedb(edisgo_object=self,
                               **kwargs)

    def analyze(self, mode=None, timesteps=None):
        """Conducts a static, non-linear power flow analysis

        Conducts a static, non-linear power flow analysis using
        `PyPSA <https://www.pypsa.org/doc/power_flow.html#full-non-linear-power-flow>`_
        and writes results (active, reactive and apparent power as well as
        current on lines and voltages at buses) to
        :class:`~.network.results.Results`
        (e.g. :attr:`~.network.results.Results.v_res` for voltages).
        See :func:`~.tools.pypsa_io.to_pypsa` for more information.

        Parameters
        ----------
        mode : str
            Allows to toggle between power flow analysis (PFA) on the whole
            network topology (default: None), only MV ('mv' or 'mvlv') or only
            LV ('lv'). Defaults to None which equals power flow analysis for
            MV + LV.
        timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
            :pandas:`pandas.Timestamp<Timestamp>`
            Timesteps specifies for which time steps to conduct the power flow
            analysis. It defaults to None in which case the time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex` are
            used.

        """
        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        pypsa_network = self.to_pypsa(mode=mode, timesteps=timesteps)

        # run power flow analysis
        pf_results = pypsa_network.pf(timesteps)

        if all(pf_results["converged"]["0"].tolist()):
            pypsa_io.process_pfa_results(self, pypsa_network, timesteps)
        else:
            raise ValueError("Power flow analysis did not converge for the"
                             "following time steps: {}.".format(
                timesteps[~pf_results["converged"]["0"]].tolist())
            )

    def reinforce(self, **kwargs):
        """
        Reinforces the network and calculates network expansion costs.

        See :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid` for more
        information.

        """
        results = reinforce_grid(
            self,
            max_while_iterations=kwargs.get("max_while_iterations", 10),
            copy_graph=kwargs.get("copy_graph", False),
            timesteps_pfa=kwargs.get("timesteps_pfa", None),
            combined_analysis=kwargs.get("combined_analysis", False),
            mode=kwargs.get("mode", None),
        )

        # add measure to Results object
        if not kwargs.get("copy_graph", False):
            self.results.measures = "grid_expansion"

        return results

    def perform_mp_opf(self, timesteps, storage_series=[], **kwargs):
        """
        Run optimal power flow with julia.

        Parameters
        -----------
        timesteps : list
            List of timesteps to perform OPF for.
        kwargs :
            See :func:`~.opf.run_mp_opf.run_mp_opf` for further
            information.

        Returns
        --------
        str
            Status of optimization.

        """
        status = run_mp_opf(
            self, timesteps, storage_series=storage_series, **kwargs
        )
        return status

    def plot_mv_grid_topology(self, technologies=False, **kwargs):
        """
        Plots plain MV network topology and optionally nodes by technology type
        (e.g. station or generator).

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        Parameters
        ----------
        technologies : :obj:`Boolean`
            If True plots stations, generators, etc. in the topology in
            different colors. If False does not plot any nodes. Default: False.

        """

        plots.mv_grid_topology(
            self,
            node_color="technology" if technologies is True else None,
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            title=kwargs.get("title", ""),
        )

    def plot_mv_voltages(self, **kwargs):
        """
        Plots voltages in MV network on network topology plot.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.v_res is None:
                logging.warning(
                    "Voltages from power flow "
                    "analysis must be available to plot them."
                )
                return
        except AttributeError:
            logging.warning(
                "Results must be available to plot voltages. "
                "Please analyze grid first."
            )
            return
        except ValueError:
            pass

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            node_color="voltage",
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            voltage=self.results.v_res,
            limits_cb_nodes=kwargs.get("limits_cb_nodes", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            title=kwargs.get("title", ""),
        )

    def plot_mv_line_loading(self, **kwargs):
        """
        Plots relative line loading (current from power flow analysis to
        allowed current) of MV lines.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.i_res is None:
                logging.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot line loading."
                )
                return
        except AttributeError:
            logging.warning(
                "Results must be available to plot line loading. "
                "Please analyze grid first."
            )
            return

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            line_color="loading",
            node_color=kwargs.get("node_color", None),
            line_load=self.results.i_res,
            filename=kwargs.get("filename", None),
            arrows=kwargs.get("arrows", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            voltage=self.results.v_res,
            limits_cb_lines=kwargs.get("limits_cb_lines", None),
            limits_cb_nodes=kwargs.get("limits_cb_nodes", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            lines_cmap=kwargs.get("lines_cmap", "inferno_r"),
            title=kwargs.get("title", ""),
            scaling_factor_line_width=kwargs.get(
                "scaling_factor_line_width", None
            ),
            curtailment_df=kwargs.get("curtailment_df", None),
        )

    def plot_mv_grid_expansion_costs(self, **kwargs):
        """
        Plots grid expansion costs per MV line.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.grid_expansion_costs is None:
                logging.warning(
                    "Grid expansion cost results needed to plot "
                    "them. Please do grid reinforcement."
                )
                return
        except AttributeError:
            logging.warning(
                "Results of MV topology needed to  plot topology "
                "expansion costs. Please reinforce first."
            )
            return

        plots.mv_grid_topology(
            self,
            line_color="expansion_costs",
            grid_expansion_costs=self.results.grid_expansion_costs,
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            limits_cb_lines=kwargs.get("limits_cb_lines", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            lines_cmap=kwargs.get("lines_cmap", "inferno_r"),
            title=kwargs.get("title", ""),
            scaling_factor_line_width=kwargs.get(
                "scaling_factor_line_width", None
            ),
        )

    def plot_mv_storage_integration(self, **kwargs):
        """
        Plots storage position in MV topology of integrated storage units.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        plots.mv_grid_topology(
            self, node_color="storage_integration", **kwargs
        )

    def plot_mv_grid(self, **kwargs):
        """
        General plotting function giving all options of function
        :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        plots.mv_grid_topology(self, **kwargs)

    def histogram_voltage(self, timestep=None, title=True, **kwargs):
        """
        Plots histogram of voltages.

        For more information on the histogram plot and possible configurations
        see :func:`edisgo.tools.plots.histogram`.

        Parameters
        ----------
        timestep : :pandas:`pandas.Timestamp<Timestamp>` or list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
            Specifies time steps histogram is plotted for. If timestep is None
            all time steps voltages are calculated for are used. Default: None.
        title : :obj:`str` or :obj:`bool`, optional
            Title for plot. If True title is auto generated. If False plot has
            no title. If :obj:`str`, the provided title is used. Default: True.

        """
        try:
            data = self.results.v_res
            if data is None:
                logger.warning(
                    "Results for voltages are required for "
                    "voltage histogramm. Please analyze first."
                )
                return
        except AttributeError:
            logger.warning(
                "Results are required for "
                "voltage histogramm. Please analyze first."
            )
            return

        if timestep is None:
            timestep = data.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]

        if title is True:
            if len(timestep) == 1:
                title = "Voltage histogram for time step {}".format(
                    timestep[0]
                )
            else:
                title = "Voltage histogram \nfor time steps {} to {}".format(
                    timestep[0], timestep[-1]
                )
        elif title is False:
            title = None
        plots.histogram(data=data, title=title, timeindex=timestep, **kwargs)

    def histogram_relative_line_load(
        self, timestep=None, title=True, voltage_level="mv_lv", **kwargs
    ):
        """
        Plots histogram of relative line loads.

        For more information on how the relative line load is calculated see
        :func:`edisgo.tools.tools.get_line_loading_from_network`.
        For more information on the histogram plot and possible configurations
        see :func:`edisgo.tools.plots.histogram`.

        Parameters
        ----------
        timestep : :pandas:`pandas.Timestamp<Timestamp>` or list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
            Specifies time step(s) histogram is plotted for. If `timestep` is
            None all time steps currents are calculated for are used.
            Default: None.
        title : :obj:`str` or :obj:`bool`, optional
            Title for plot. If True title is auto generated. If False plot has
            no title. If :obj:`str`, the provided title is used. Default: True.
        voltage_level : :obj:`str`
            Specifies which voltage level to plot voltage histogram for.
            Possible options are 'mv', 'lv' and 'mv_lv'. 'mv_lv' is also the
            fallback option in case of wrong input. Default: 'mv_lv'

        """
        try:
            if self.results.i_res is None:
                logger.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot histogram line "
                    "loading."
                )
                return
        except AttributeError:
            logger.warning(
                "Results must be available to plot histogram line "
                "loading. Please analyze grid first."
            )
            return

        if voltage_level == "mv":
            lines = self.topology.mv_grid.lines_df
        elif voltage_level == "lv":
            lines = self.topology.lines_df[
                ~self.topology.lines_df.index.isin(
                    self.topology.mv_grid.lines_df.index)
            ]
        else:
            lines = self.topology.lines_df

        rel_line_loading = tools.calculate_relative_line_load(
            self, lines.index, timestep
        )

        if timestep is None:
            timestep = rel_line_loading.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]

        if title is True:
            if len(timestep) == 1:
                title = "Relative line load histogram for time step {}".format(
                    timestep[0]
                )
            else:
                title = (
                    "Relative line load histogram \nfor time steps "
                    "{} to {}".format(timestep[0], timestep[-1])
                )
        elif title is False:
            title = None
        plots.histogram(data=rel_line_loading, title=title, **kwargs)

    def save(
        self,
        directory,
        save_results=True,
        save_topology=True,
        save_timeseries=True,
        **kwargs
    ):
        """
        Saves edisgo_obj parameters to csv. It can be chosen if results,
        topology and timeseries should be saved, respectively. For each one, a
        separate folder is created.

        Parameters
        ----------
        directory: str
            directory to save edisgo_obj to. Subfolders for respective
            parameters will be created.
        save_results: bool
            indicates whether to save self.results
        save_topology: bool
            indicates whether to save self.topology
        save_timeseries: bool
            indicates whether to save self.timeseries

        """
        os.makedirs(directory, exist_ok=True)
        if save_results:
            results_dir = os.path.join(directory, "results")
            os.makedirs(results_dir, exist_ok=True)
            kwargs.get("parameters", "all")
            self.results.save(results_dir)
        if save_topology:
            topology_dir = os.path.join(directory, "topology")
            self.topology.to_csv(topology_dir)
        if save_timeseries:
            self.timeseries.to_csv(directory)

    def add_component(
        self,
        comp_type,
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs
    ):
        """
        Adds single component to topology and respective timeseries if add_ts
        is set to True.

        Parameters
        ----------
        comp_type: str
            Type of added component. Can be 'Bus', 'Line', 'Load', 'Generator',
            'StorageUnit', 'Transformer'
        add_ts: Boolean
            Indicator if time series for component are added as well
        ts_active_power : pd.Series
            Active power time series of added component.
        ts_reactive_power : pd.Series
            Reactive power time series of added component.
        **kwargs: dict
            Attributes of added component. See respective functions for required
            entries. For 'Load', 'Generator' and 'StorageUnit' the boolean
            add_ts determines whether a time series is created for the new
            component or not.

        Todo: change into add_components to allow adding of several components
            at a time, change topology.add_load etc. to add_loads, where
            lists of parameters can be inserted
        """
        if comp_type == "Bus":
            self.topology.add_bus(bus_name=kwargs.get("name"), **kwargs)
            comp_name = kwargs.get("name")
        elif comp_type == "Line":
            comp_name = self.topology.add_line(**kwargs)
        elif comp_type == "Load" or comp_type == "charging_park":
            comp_name = self.topology.add_load(
                load_id=kwargs.get("load_id"),
                bus=kwargs.get("bus"),
                peak_load=kwargs.get("peak_load"),
                annual_consumption=kwargs.get("annual_consumption"),
                sector=kwargs.get("sector"),
            )
            if add_ts:
                timeseries.add_loads_timeseries(
                    edisgo_obj=self, load_names=comp_name, **kwargs
                )

        elif comp_type == "Generator":
            comp_name = self.topology.add_generator(**kwargs)
            if add_ts:
                timeseries.add_generators_timeseries(
                    edisgo_obj=self, generator_names=comp_name, **kwargs
                )

        elif comp_type == "ChargingPoint":
            comp_name = self.topology.add_charging_point(**kwargs)
            if add_ts:
                if ts_active_power is not None and ts_reactive_power is not None:
                    timeseries.add_charging_points_timeseries(
                        self, [comp_name],
                        ts_active_power=pd.DataFrame({
                            comp_name: ts_active_power}),
                        ts_reactive_power=pd.DataFrame({
                            comp_name: ts_reactive_power})
                    )
                else:
                    raise ValueError("Time series for charging points need "
                                     "to be provided.")

        elif comp_type == "StorageUnit":
            comp_name = self.topology.add_storage_unit(
                **kwargs,
            )
            if add_ts:
                if isinstance(ts_active_power, pd.Series):
                    ts_active_power = pd.DataFrame(
                        {comp_name: ts_active_power}
                    )
                if isinstance(ts_reactive_power, pd.Series):
                    ts_reactive_power = pd.DataFrame(
                        {comp_name: ts_reactive_power}
                    )
                timeseries.add_storage_units_timeseries(
                    edisgo_obj=self,
                    storage_unit_names=comp_name,
                    timeseries_storage_units=ts_active_power,
                    timeseries_storage_units_reactive_power=ts_reactive_power,
                    **kwargs
                )
        else:
            raise ValueError("Component type is not correct.")
        return comp_name

    def integrate_component(
        self,
        comp_type,
        geolocation,
        voltage_level=None,
        mode="mv",
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs
    ):
        """
        Adds single component to topology based on geolocation.

        This function integrates a new generator or charging point into the
        grid.

        Parameters
        ----------
        comp_type: str
            Type of added component. Can be 'Generator' or 'ChargingPoint'.
        geolocation : tuple or shapely Point
            Geolocation of the new component. In case of tuple, the geolocation
            must be given in the form (longitude, latitude).
        voltage_level : int
            Specifies the voltage level the new component is integrated in.
            Possible options are 4 (MV busbar), 5 (MV grid), 6 (LV busbar) or
            7 (LV grid). If no voltage level is provided the voltage level
            is determined based on the nominal power `p_nom` (given as kwarg)
            as follows:

            * voltage level 4 (MV busbar): nominal power between 4.5 MW and
              17.5 MW
            * voltage level 5 (MV grid) : nominal power between 0.3 MW and
              4.5 MW
            * voltage level 6 (LV busbar): nominal power between 0.1 MW and
              0.3 MW
            * voltage level 7 (LV grid): nominal power below 0.1 MW

        mode : str
            So far, only mode "mv" is implemented. In that case, components
            in the low voltage will be connected to the closest substation's
            secondary side.
        add_ts: Boolean
            Indicator if time series for component are added as well
        ts_active_power : pd.Series
            Active power time series of added component.
        ts_reactive_power : pd.Series
            Reactive power time series of added component.
        **kwargs: dict
            Attributes of added component. See respective functions for required
            entries.

        """

        supported_voltage_levels = set([4, 5, 6, 7])
        p_nom = kwargs.get('p_nom', None)
        if not voltage_level in supported_voltage_levels:
            if p_nom is None:
                raise ValueError(
                    "Neither appropriate voltage level nor nominal power "
                    "were supplied.")
            # Calculate voltage level manually from nominal power:
            if p_nom > 4.5 and p_nom <= 17.5:
                voltage_level = 4
            elif p_nom > 0.3 and p_nom <= 4.5:
                voltage_level = 5
            elif p_nom > 0.1 and p_nom <= 0.3:
                voltage_level = 6
            elif p_nom > 0 and p_nom <= 0.1:
                voltage_level = 7
            else:
                raise ValueError("Unsupported voltage level")

        # check if geolocation is given as shapely Point, otherwise transform
        # to shapely Point
        if not type(geolocation) is Point:
            geolocation = Point(geolocation)

        # Connect MV component
        if voltage_level == 4:
            self.add_component(comp_type,
                               bus=self.topology.mv_grid.station.index[0],
                               add_ts=add_ts,
                               ts_active_power=ts_active_power,
                               ts_reactive_power=ts_reactive_power, **kwargs)
        elif voltage_level == 5:
            # Create common properties for Generator and Charging Point
            properties = pd.Series()
            properties['voltage_level'] = voltage_level
            properties['geom'] = str(geolocation)
            properties['electrical_capacity'] = p_nom
            properties.name = kwargs.get('generator_id', None)
            properties['generation_type'] = kwargs.get('generator_type', None)
            properties['generation_subtype'] = kwargs.get(
                'generator_subtype', None)
            properties['w_id'] = kwargs.get('weather_cell_id', None)
            comp_name = add_and_connect_mv_generator(
                self, properties, comp_type)

            if add_ts and comp_type == 'Generator':
                timeseries.add_generators_timeseries(
                    edisgo_obj=self, generator_names=comp_name, **kwargs
                )
            elif add_ts and comp_type == 'ChargingPoint':
                timeseries.add_charging_points_timeseries(
                    self, [comp_name],
                    ts_active_power=pd.DataFrame({
                        comp_name: ts_active_power}),
                    ts_reactive_power=pd.DataFrame({
                        comp_name: ts_reactive_power})
                )

        # Connect LV component
        else:
            substations = self.topology.buses_df.loc[
                self.topology.transformers_df.bus1]
            nearest_substation, _ = find_nearest_bus(geolocation, substations)
            self.add_component(comp_type,
                               bus=nearest_substation,
                               add_ts=add_ts,
                               ts_active_power=ts_active_power,
                               ts_reactive_power=ts_reactive_power, **kwargs)

    def remove_component(self, comp_type, comp_name, drop_ts=True):
        """
        Removes single component from respective DataFrame. If drop_ts is set
        to True, timeseries of elements are deleted as well.

        Parameters
        ----------
        comp_type: str
            Type of removed component. Can be 'Bus', 'Line', 'Load',
            'Generator', 'StorageUnit', 'Transformer'.
        comp_name: str
            Name of component to be removed.
        drop_ts: Boolean
            Indicator if timeseries for component are removed as well. Defaults
            to True.

        Todo: change into remove_components, when add_component is changed into
            add_components, to allow removal of several components at a time

        """
        if comp_type == "Bus":
            self.topology.remove_bus(comp_name)
        elif comp_type == "Line":
            self.topology.remove_line(comp_name)
        elif comp_type == "Load":
            self.topology.remove_load(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self, comp_type="loads", comp_names=comp_name
                )

        elif comp_type == "Generator":
            self.topology.remove_generator(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self,
                    comp_type="generators",
                    comp_names=comp_name,
                )
        elif comp_type == "StorageUnit":
            self.topology.remove_storage(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self,
                    comp_type="storage_units",
                    comp_names=comp_name,
                )
        elif comp_type == "ChargingPoint":
            self.topology.remove_charging_point(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self,
                    comp_type="charging_points",
                    comp_names=comp_name
                )
        else:
            raise ValueError("Component type is not correct.")
