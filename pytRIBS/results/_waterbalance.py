import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_mrf_water_balance(instance, method):
    """
    """

    drainage_area = instance.watershed_stats['area'] = instance.voronoi['geometry'].area.sum()
    porosity = instance.int_spatial_vars['Bedrock_Depth_mm'].mean()
    mrf = instance.mrf

    if any([drainage_area, porosity, mrf]) is None:
        print('Drainage Area, Porosity, or MRF file is None.\n Unable to calculate water balance from MRF File.')
        return

    waterbalance = _run_mrf_water_balance(mrf, porosity, drainage_area, method)
    var = {"mrf": mrf, "waterbalance": waterbalance}
    instance.mrf = var


def get_element_water_balance(instance, method, node_file=None):
    """
    This function loops through element_results and assigns water_balance as second item in the list for a given
    node/key. The user can specify a method for calculating the time frames over which the water balance is
    calculated.
    """
    # read in node list
    if node_file is None:
        node_file = instance.options["nodeoutputlist"]["value"]

    nodes = instance.read_node_list(node_file)
    invar_data_frame = instance.element['invar']

    for n in nodes:
        n = int(n)
        porosity = invar_data_frame.Porosity[invar_data_frame.NodeID == n].values[0]
        element_area = invar_data_frame.Area_m_sq[invar_data_frame.NodeID == n].values[0]
        waterbalance = instance.run_element_water_balance(instance.element[n], porosity, element_area, method)
        instance.element.update({n: {"pixel": instance.element[n], "waterbalance": waterbalance}})


# WATER BALANCE FUNCTIONS
def run_element_water_balance(instance, data, porosity, element_area, method):
    """

    :param data:
    :param porosity:
    :param element_area:
    :param method:
    :return: pandas data frame with waterbalance information
    """

    begin, end, timeframe = instance.water_balance_dates(data.Time, method)

    for n in range(0, len(timeframe)):
        if n == 0:
            waterbalance = instance.element_wb_components(data, begin[n], end[n], porosity, element_area)
        else:
            temp = instance.element_wb_components(data, begin[n], end[n], porosity, element_area)

            for key, val in temp.items():

                if key in waterbalance:
                    waterbalance[key] = np.append(waterbalance[key], val)

    waterbalance.update({"Time": timeframe})
    waterbalance = pd.DataFrame.from_dict(waterbalance)
    waterbalance.set_index('Time', inplace=True)

    waterbalance['dS'] = np.add(np.add(waterbalance['dUnsat'], waterbalance['dSat'], waterbalance['dCanopySWE']),
                                waterbalance['dSWE'],
                                waterbalance['dCanopy'])  # change in storage

    waterbalance['nQ'] = np.add(waterbalance['nQsurf'], waterbalance['nQunsat'],
                                waterbalance['nQsat'])  # net fluxes from surface and saturated and unsaturated zone

    return waterbalance


@staticmethod
def element_wb_components(element_data_frame, begin, end, porosity, element_area):
    """
    Computes water balance calculations for an individual computational element or node over a specified time frame. Data = pandas data
    frame of .pixel file, begin is start date, end is end date, bedrock depth is the depth to bedrock,
    porosity is well, porosity, and element area is surface area of voronoi polygon. Returns a dictionary with
    individual water components, keys with the prescript d indicate change in storage (i.e. delta) and n
    indicates net cumulative flux.
    """

    # logical index for calculating water balance
    begin_id = element_data_frame['Time'].values == begin
    end_id = element_data_frame['Time'].values == end
    duration_id = (element_data_frame['Time'].values >= begin) & (
            element_data_frame['Time'].values <= end)

    # return dictionary with values
    waterbalance = {}

    # Store ET flux as series due to complexity
    evapotrans = element_data_frame['EvpTtrs_mm_h'] - (
            element_data_frame.SnSub_cm * 10 + element_data_frame.SnEvap_cm * 10 + element_data_frame.IntSub_cm * 10)  # Snow evaporation fluxes are subtracted due to signed behavior in snow module

    # calculate individual water balance components
    waterbalance.update(
        {'dUnsat': element_data_frame.Mu_mm.values[end_id][0] - element_data_frame.Mu_mm.values[begin_id][
            0]})  # [0] converts from array to float
    waterbalance.update(
        {'dSat': (element_data_frame.Nwt_mm.values[begin_id][0] - element_data_frame.Nwt_mm.values[end_id][
            0]) * porosity})
    waterbalance.update({'dCanopySWE': (10 * (
            element_data_frame.IntSWEq_cm.values[end_id][0] - element_data_frame.IntSWEq_cm.values[begin_id][
        0]))})  # convert from cm to mm
    waterbalance.update({'dSWE': (10 * (
            element_data_frame.SnWE_cm.values[end_id][0] - element_data_frame.SnWE_cm.values[begin_id][0]))})
    waterbalance.update({'dCanopy': element_data_frame.CanStorage_mm.values[end_id][0] -
                                    element_data_frame.CanStorage_mm.values[begin_id][0]})
    waterbalance.update({'nP': np.sum(element_data_frame['Rain_mm_h'].values[duration_id])})
    waterbalance.update({'nET': np.sum(evapotrans.values[duration_id])})
    waterbalance.update({'nQsurf': np.sum(element_data_frame['Srf_Hour_mm'].values[duration_id])})
    waterbalance.update(
        {'nQunsat': np.sum(element_data_frame['QpIn_mm_h'].values[duration_id]) - np.sum(
            element_data_frame['QpOut_mm_h'].values[duration_id])})
    waterbalance.update(
        {'nQsat': np.sum(
            element_data_frame['GWflx_m3_h'].values[
                duration_id]) / element_area * 1000})  # convert from m^3/h to mm/h

    return waterbalance


def _run_mrf_water_balance(data, porosity, drainage_area, method):
    """

    :param data:
    :param porosity:
    :param method:
    :return: pandas data frame with waterbalance information
    """
    global waterbalance

    if method == 'full':
        min_date = min(data.Time)
        max_date = max(data.Time)
        waterbalance = _mrf_wb_components(data, min_date, max_date, porosity, bedrock, drainage_area)
        timeframe = data.Time.mean()
    else:
        begin, end, timeframe = _water_balance_dates(data.Time, method)

        for n in range(0, len(timeframe)):
            if n == 0:
                waterbalance = _mrf_wb_components(data, begin[n], end[n], porosity, drainage_area)
            else:
                temp = _mrf_wb_components(data, begin[n], end[n], porosity, drainage_area)

                for key, val in temp.items():

                    if key in waterbalance:
                        waterbalance[key] = np.append(waterbalance[key], val)

    waterbalance.update({"Time": timeframe})
    # change in storage
    waterbalance.update({"dS": waterbalance['dUnsat'] + waterbalance['dSat'] + waterbalance['dCanopySWE'] +
                               waterbalance['dSWE'] + waterbalance['dCanopy']})
    # net fluxes from surface and saturated and unsaturated zone
    waterbalance.update({'nQ': waterbalance['nQsurf'] + waterbalance['nQunsat'] + waterbalance['nQsat']})

    waterbalance = pd.DataFrame.from_dict(waterbalance)
    waterbalance.set_index('Time', inplace=True)

    return waterbalance


@staticmethod
def _mrf_wb_components(mrf_data_frame, begin, end, porosity, drainage_area):
    """
    Computes water balance calculations for an individual computational mrf or node over a specified time frame. Data = pandas data
    frame of .pixel file, begin is start date, end is end date, bedrock depth is the depth to bedrock,
    porosity is well, porosity, and mrf area is surface area of voronoi polygon. Returns a dictionary with
    individual water components, keys with the prescript d indicate change in storage (i.e. delta) and n
    indicates net cumulative flux.
    """

    # logical index for calculating water balance
    begin_id = mrf_data_frame['Time'].values == begin
    end_id = mrf_data_frame['Time'].values == end
    duration_id = (mrf_data_frame['Time'].values >= begin) & (
            mrf_data_frame['Time'].values <= end)

    # return dictionary with values
    waterbalance = {}

    # Store ET flux as series due to complexity
    # Snow evaporation fluxes are subtracted due to signed behavior in snow module
    evapotrans = mrf_data_frame.MET - 10 * (
            mrf_data_frame.AvSnSub + mrf_data_frame.AvSnEvap + mrf_data_frame.AvInSu)
    unsaturated = mrf_data_frame.MSMU.values * porosity * mrf_data_frame.MDGW.values
    # calculate individual water balance components
    waterbalance.update(
        {'dSat': (mrf_data_frame.MDGW.values[begin_id] - mrf_data_frame.MDGW.values[end_id]) * porosity})
    waterbalance.update(
        {'dUnsat': (unsaturated[begin_id] - unsaturated[end_id])})
    waterbalance.update({'dCanopySWE': (10 * (
            mrf_data_frame.AvInSn.values[end_id][0] - mrf_data_frame.AvInSn.values[begin_id][
        0]))})  # convert from cm to mm
    waterbalance.update({'dSWE': (10 * (
            mrf_data_frame.AvSWE.values[end_id][0] - mrf_data_frame.AvSWE.values[begin_id][0]))})
    waterbalance.update({'dCanopy': 0})  # TODO update mrf w/ mean intercepted canpoy storaage
    waterbalance.update({'nP': np.sum(mrf_data_frame.MAP.values[duration_id])})
    waterbalance.update({'nET': np.sum(evapotrans.values[duration_id])})
    waterbalance.update({'nQsurf': np.sum(mrf_data_frame.Srf.values[duration_id] * 3600 * 1000 / drainage_area)})
    waterbalance.update({'nQunsat': 0})  # Assumption in model is closed boundaries at divide and outled
    waterbalance.update({'nQsat': 0})

    return waterbalance


@staticmethod
def plot_water_balance(waterbalance, saved_fig=None):
    """

    :param saved_fig:
   :param waterbalance:
   :return:
   """

    # plt.style.use('bmh')
    barwidth = 0.25
    fig, ax = plt.subplots()

    ax.bar(np.arange(len(waterbalance)) + barwidth, waterbalance['nP'], align='center', width=barwidth,
           color='grey', label='nP')
    rects = ax.patches

    # Make some labels.
    labels = ["%.0f" % (p - waterbalance) for p, waterbalance in
              zip(waterbalance['nP'], waterbalance['dS'] + waterbalance['nQ'] + waterbalance['nET'])]
    netdiff = [p - waterbalance for p, waterbalance in
               zip(waterbalance['nP'], waterbalance['dS'] + waterbalance['nQ'] + waterbalance['nET'])]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )

    ax.text(len(waterbalance.index), max(waterbalance.nP), "mean difference: " + "%.0f" % np.mean(netdiff))

    waterbalance.plot.bar(ax=ax, y=["nQ", "nET", "dS"], stacked=True, width=barwidth,
                          color=['tab:blue', 'tab:red', 'tab:cyan'])
    ax.legend(bbox_to_anchor=(1.35, 0.85), loc='center right',
              labels=["Precip.", "Runoff", "Evapo. Trans.", "$\Delta$ Storage"])
    ax.set_ylabel("Water Flux & $\Delta$ Storage (mm)")
    ax.set_xticks(range(len(waterbalance.index)), waterbalance.index.strftime("%Y-%m"), rotation=45)
    fig.autofmt_xdate()
    plt.show()

    if saved_fig is not None:
        plt.savefig(saved_fig, bbox_inches='tight')

    return fig, ax


@staticmethod
def _water_balance_dates(time_vector, method):
    """
    Returns three vectors of time objects for specified time intrevals: beginning dates, ending dates,
    and years.
    :param time_vector: Vector containing sequential dates from model simulations.
    :param str method: String specifying time interval to segment time vector. "water_year", segments time frame by water year and
    discards results that do not start first or end on last water year in data. "year", just segments based on
    year, "month" segments base on month, and "cold_warm",segments on cold (Oct-April) and warm season (May-Sep).
    """

    min_date = min(time_vector)
    max_date = max(time_vector)
    begin_dates = None
    end_dates = None

    if method == "water_year":
        years = np.arange(min_date.year, max_date.year)
        begin_dates = [pd.Timestamp(year=x, month=10, day=1, hour=0, minute=0, second=0) for x in years]
        years += 1
        end_dates = [pd.Timestamp(year=x, month=9, day=30, hour=23, minute=0, second=0) for x in years]

        # make sure water years are in data set
        while begin_dates[0] < min_date:
            begin_dates.pop(0)
            end_dates.pop(0)

        while end_dates[len(end_dates) - 1] > max_date:
            begin_dates.pop(len(end_dates) - 1)
            end_dates.pop(len(end_dates) - 1)

    if method == "year":
        years = np.arange(min_date.year, max_date.year)
        begin_dates = [pd.Timestamp(year=x, month=1, day=1, hour=0, minute=0, second=0) for x in years]
        end_dates = [pd.Timestamp(year=x, month=12, day=31, hour=23, minute=0, second=0) for x in years]

        # adjust start date according to min_date
        begin_dates[0] = min_date

        # add ending date according to end_date
        end_dates.append(max_date)

        # add last year to years
        years = np.append(years, max_date.year)

    if method == "cold_warm":
        years = np.arange(min_date.year, max_date.year + 1)
        begin_dates = [[pd.Timestamp(year=x, month=5, day=1, hour=0, minute=0, second=0),
                        pd.Timestamp(year=x, month=10, day=1, hour=0, minute=0, second=0)] for x in years]
        begin_dates = [date for sublist in begin_dates for date in sublist]
        end_dates = [[pd.Timestamp(year=x, month=9, day=30, hour=23, minute=0, second=0),
                      pd.Timestamp(year=x + 1, month=4, day=30, hour=23, minute=0, second=0)] for x in years]
        end_dates = [date for sublist in end_dates for date in sublist]

        # make sure season are in data set
        while begin_dates[0] < min_date:
            begin_dates.pop(0)
            end_dates.pop(0)

        while end_dates[len(end_dates) - 1] > max_date:
            begin_dates.pop(len(end_dates) - 1)
            end_dates.pop(len(end_dates) - 1)

    # Update date time to reflect middle of period over which the waterbalance is calculated
    years = [x + (y - x) / 2 for x, y in zip(begin_dates, end_dates)]
    return begin_dates, end_dates, years
