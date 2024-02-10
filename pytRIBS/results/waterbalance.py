import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class WaterBalance:
    def get_mrf_water_balance(self, method):
        """
        """
        waterbalance = self._run_mrf_water_balance(self,method)
        self.mrf['waterbalance'] = waterbalance
    
    
    def get_element_water_balance(self, method):
        """
        Calculate water balance for elements in the model.
    
        This function iterates through element results and calculates the water balance for a specific node or key.
        The user can choose the method for calculating the time frames over which the water balance is computed.
    
        Parameters:
        - self: An self of the model containing element results.
        - method: A string specifying the calculation method ('full') or a list of custom date ranges (min and max).
    
        Example:
        1. Calculate full water balance: get_element_water_balance(self, 'full')
        2. Calculate custom water balance for specific dates: get_element_water_balance(self, ['2024-01-01', '2024-01-31'])
        """
    
        node_file = self.options["nodeoutputlist"]["value"]
        nodes = self.read_node_list(node_file)
    
        for n in nodes:
            n = int(n)
    
            if isinstance(self.element[n], dict):
                data = self.element[n]['pixel']  # waterbalance calcs have already been run
            else:
                data = self.element[n]
    
    
            waterbalance = self._run_element_water_balance(self, n, method)
            self.element[n]["waterbalance"] = waterbalance
    
    
    # WATER BALANCE FUNCTIONS
    def _run_element_water_balance(self, element_id, method):
        """
    
        :param data:
        :param porosity:
        :param element_area:
        :param method:
        :return: pandas data frame with waterbalance information
        """
        data = self.element[element_id]
        if isinstance(data, dict):
            data = data['pixel']  # waterbalance calc already called
    
        if method == 'full':
            min_date = min(data.Time)
            max_date = max(data.Time)
            waterbalance = self._element_wb_components(element_id, min_date, max_date)
            timeframe = data.Time.mean()
    
        elif isinstance(method, list):
            min_date = min(method)
            max_date = max(method)
            waterbalance = self._element_wb_components(element_id, min_date, max_date)
            timeframe = data.Time.mean()
    
        else:
            begin, end, timeframe = self._water_balance_dates(data.Time, method)
    
            for n in range(0, len(timeframe)):
                if n == 0:
                    waterbalance = self._element_wb_components(element_id, begin[n], end[n])
                else:
                    temp = self._element_wb_components(element_id, begin[n], end[n])
    
                    for key, val in temp.items():
    
                        if key in waterbalance:
                            waterbalance[key] = np.append(waterbalance[key], val)
    
        waterbalance.update({"Time": timeframe})
    
        if method == 'full' or isinstance(method, list):
            waterbalance = pd.DataFrame(waterbalance, index=[0])
        else:
            waterbalance = pd.DataFrame.from_dict(waterbalance)
    
        waterbalance.set_index('Time', inplace=True)
    
        waterbalance['dS'] = waterbalance['dUnsat'].values + waterbalance['dSat'].values + waterbalance['dCanopySWE'].values \
                             + waterbalance['dSWE'].values + waterbalance['dCanopy'].values
    
        # net fluxes from surface and saturated and unsaturated zone
        waterbalance['nQ'] = waterbalance['nQsurf'].values + waterbalance['nQunsat'].values + waterbalance['nQsat'].values
    
        return waterbalance
    
    

    def _element_wb_components(self, element_id, begin, end):
        """
        Computes water balance calculations for an individual computational element or node over a specified time frame. Data = pandas data
        frame of .pixel file, begin is start date, end is end date, bedrock depth is the depth to bedrock,
        porosity is well, porosity, and element area is surface area of voronoi polygon. Returns a dictionary with
        individual water components, keys with the prescript d indicate change in storage (i.e. delta) and n
        indicates net cumulative flux.
        """
    
        # data frame with continuous water balance components
        df = self.get_element_wb_dataframe(element_id)
        duration_id = (df['Time'].values >= begin) & (df['Time'].values <= end)
    
        df = df[duration_id]
    
        # return dictionary with values
        waterbalance = {}
    
        # calculate individual water balance components
        waterbalance.update({'dUnsat': df['Unsat_mm'].iloc[-1] - df['Unsat_mm'].iloc[0]})
        waterbalance.update({'dSat': df['Sat_mm'].iloc[0] - df['Sat_mm'].iloc[-1]})
        waterbalance.update({'dCanopySWE': df['CanopySWE_mm'].iloc[-1] - df['CanopySWE_mm'].iloc[0]})
        waterbalance.update({'dSWE': df['SWE_mm'].iloc[-1] - df['SWE_mm'].iloc[0]})
        waterbalance.update({'dCanopy': df['Canop_mm'].iloc[-1] - df['Canop_mm'].iloc[0]})
        waterbalance.update({'nP': df['P_mm_h'].sum()})
        waterbalance.update({'nET': df['ET_mm_h'].sum()})
        waterbalance.update({'nQsurf': df['Qsurf_mm_h'].sum()})
        waterbalance.update({'nQunsat': df['Qunsat_mm_h'].sum()})
        waterbalance.update({'nQsat': df['Qsat_mm_h'].sum()})
    
        return waterbalance
    
    
    def _run_mrf_water_balance(self, method):
        """
    
        :param data:
        :param porosity:
        :param method:
        :return: pandas data frame with waterbalance information
        """
        global waterbalance
        data = self.mrf['mrf']
    
        if method == 'full':
            min_date = min(data.Time)
            max_date = max(data.Time)
            waterbalance = self._mrf_wb_components(min_date, max_date)
            timeframe = data.Time.mean()
    
        elif isinstance(method, list):
            min_date = min(method)
            max_date = max(method)
            waterbalance = self._mrf_wb_components(min_date, max_date)
            timeframe = data.Time.mean()
    
        else:
            begin, end, timeframe = self._water_balance_dates(data.Time, method)
    
            for n in range(0, len(timeframe)):
                if n == 0:
                    waterbalance = self._mrf_wb_components(begin[n], end[n])
                else:
                    temp = self._mrf_wb_components(begin[n], end[n])
    
                    for key, val in temp.items():
    
                        if key in waterbalance:
                            waterbalance[key] = np.append(waterbalance[key], val)
    
        waterbalance.update({"Time": timeframe})
        # change in storage
        waterbalance.update({"dS": waterbalance['dUnsat'] + waterbalance['dSat'] + waterbalance['dCanopySWE'] +
                                   waterbalance['dSWE'] + waterbalance['dCanopy']})
        # net fluxes from surface and saturated and unsaturated zone
        waterbalance.update({'nQ': waterbalance['nQsurf'] + waterbalance['nQunsat'] + waterbalance['nQsat']})
    
        if method == 'full' or isinstance(method, list):
            waterbalance = pd.DataFrame(waterbalance, index=[0])
        else:
            waterbalance = pd.DataFrame.from_dict(waterbalance)
    
        waterbalance.set_index('Time', inplace=True)
    
        return waterbalance
    
    

    def _mrf_wb_components(self, begin, end):
        """
        Computes water balance calculations for an individual computational mrf or node over a specified time frame. Data = pandas data
        frame of .pixel file, begin is start date, end is end date, bedrock depth is the depth to bedrock,
        porosity is well, porosity, and mrf area is surface area of voronoi polygon. Returns a dictionary with
        individual water components, keys with the prescript d indicate change in storage (i.e. delta) and n
        indicates net cumulative flux.
        """
    
        # logical index for calculating water balance
        df = self.get_mrf_wb_dataframe()
        duration_id = (df['Time'].values >= begin) & (df['Time'].values <= end)
        df = df[duration_id]
    
        # return dictionary with values
        waterbalance = {}
    
        waterbalance.update({'dSat': df['Sat_mm'].iloc[0]-df['Sat_mm'].iloc[-1]})
        waterbalance.update({'dUnsat': df['Unsat_mm'].iloc[-1] - df['Unsat_mm'].iloc[0]})
        waterbalance.update({'dCanopySWE': df['CanopySWE_mm'].iloc[-1] - df['CanopySWE_mm'].iloc[0]})  # convert from cm to mm
        waterbalance.update({'dSWE': df['SWE_mm'].iloc[-1] - df['SWE_mm'].iloc[0]})
        waterbalance.update({'dCanopy': 0})  # TODO update mrf w/ mean intercepted canpoy storaage
        waterbalance.update({'nP': df['P_mm_h'].sum()})
        waterbalance.update({'nET': df['ET_mm_h'].sum()})
        waterbalance.update({'nQsurf': df['Qsurf_mm_h'].sum()})
        waterbalance.update({'nQunsat': 0})  # Assumption in model is closed boundaries at divide and outlet
        waterbalance.update({'nQsat': 0})
    
        return waterbalance
    
    
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
    
    
