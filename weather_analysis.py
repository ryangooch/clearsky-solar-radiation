#!/usr/bin/env python3

"""
This script models the "clear sky" solar radiation and compares it to measured solar radiation at
a weather station. We specifically look at the NOAA weather station in Boulder, CO.

The clear sky model assumes that the elevation is sea level and a simple atmospheric model as a
function of solar elevation angle of incidence. As a result, the values will underestimate high
solar radiation days.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("./data")
WEATHER_DATA_PATH = DATA_PATH / "CRNH0203-2024-CO_Boulder_14_W.txt"
HEADERS_PATH = DATA_PATH / "headers.txt"


def get_headers() -> List[str]:
    """
    Get the headers from the NOAA headers file, and format them into a comma separated list
    """
    with open(HEADERS_PATH, "r") as fil:
        lines = fil.read()
    headers_line = lines.split("\n")[1]
    # the last item is an empty string, irrelevant
    headers_names = headers_line.split(" ")[:-1]
    return headers_names


def get_data_list() -> List[List[str]]:
    """
    Get the weather station data, and format into a list of lists
    """
    with open(WEATHER_DATA_PATH, "r") as fil:
        weather_data_str: str = fil.read()

    weather_data_array = weather_data_str.split("\n")
    weather_data_list = []

    for line in weather_data_array:
        weather_data_list.append(line.split())

    return weather_data_list[:-1]


def get_data() -> pd.DataFrame:
    """
    Returns a pandas dataframe of weather station data with named columns
    """
    headers_names = get_headers()
    data_list = get_data_list()

    df = pd.DataFrame(data_list, columns=headers_names)

    return df


def plot_hourly_variable(df: pd.DataFrame, variable: str) -> None:
    """
    Plots the hourly variable values. Simple utility for interactive exploration
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(df[variable])

    ax.set_xlabel("hours elapsed")
    ax.set_ylabel(variable)
    ax.set_title(f"hourly values for {variable}")

    plt.tight_layout()
    plt.show()


def get_daily_temp_ranges(array: pd.Series) -> np.array:
    """
    Efficiently computes the daily temperature ranges for given hourly temperature data
    """
    # reshape our 1-D array into (N % 24, 24)
    len_array = len(array)

    full_days = len_array // 24
    end_index = full_days * 24

    array = array.copy()
    array = array[:end_index]
    day_matrix = np.reshape(array, (24, full_days), "F")

    return np.nanmax(day_matrix, axis=0) - np.nanmin(day_matrix, axis=0)


def get_rolling_average(array: np.array, samples: int = 24) -> np.array:
    """
    Efficiently returns the rolling average for a time series array using np.convolve
    """
    # neat way to do this is to use np.convolve
    kernel = np.ones(samples)  # our moving average window
    return np.convolve(kernel / samples, array, mode="valid")


def calculate_solar_position_series(
    timestamps: pd.Series, latitude: float, longitude: float
) -> pd.DataFrame:
    """
    where that sun at bro? This is the workhorse astronomical trig function that computes the
    solar location, in addition to a few other parameters, and returns them in a time series
    dataframe.

    the clear sky radiation model is based on the Sun-Earth system and uses the apparent position of
    the Sun in the sky based on its relative positioning by time of year (a function of latitude).
    we assume a simple atmospheric thickness model and calculate the clear sky radiation using
    calculated solar declination angle, solar azimuth angle, solar elevation angle, then clip it
    so that nighttime values are zero
    """
    timestamps = pd.to_datetime(timestamps)
    day_of_year = timestamps.dt.dayofyear.values

    # angle of sun by hour (rad)
    hour = timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600
    hour_angle = np.radians(15 * (hour - 12))

    lat_rad = np.radians(latitude)
    # don't actually need longitude for this one
    # lon_rad = np.radians(longitude)

    # solar declination angle (delta)
    # formula from Cooper (1969)
    declination = np.radians(23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81))))

    # solar elevation angle (alpha)
    sin_elevation = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(
        declination
    ) * np.cos(hour_angle)
    elevation = np.degrees(np.arcsin(sin_elevation))

    # solar azimuth angle (A)
    cos_elevation = np.cos(np.arcsin(sin_elevation))
    sin_azimuth = -np.cos(declination) * np.sin(hour_angle) / cos_elevation
    cos_azimuth = (np.sin(declination) - sin_elevation * np.sin(lat_rad)) / (
        cos_elevation * np.cos(lat_rad)
    )
    azimuth = np.degrees(np.arctan2(sin_azimuth, cos_azimuth))

    # simple theoretical clear sky radiation
    atmosphere_thickness = 1.0 / sin_elevation
    atmosphere_thickness = np.clip(atmosphere_thickness, 1.0, 38.0)  # limit extrema

    # clear sky radiation at sea level (approximation)
    solar_constant = 1361.0  # Solar constant in W/m^2
    clearsky_radiation = solar_constant * sin_elevation * 0.7**atmosphere_thickness
    clearsky_radiation = np.maximum(
        clearsky_radiation, 0
    )  # negative values don't make physical sense

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "elevation": elevation,
            "azimuth": azimuth,
            "declination": np.degrees(declination),
            "clearsky_radiation": clearsky_radiation,
            "is_daytime": elevation > 0,
        }
    )

    return result


def get_timestamps_from_concat(df: pd.DataFrame) -> pd.Series:
    """
    very specific function, handles the YYYYMMDD and HHMM column formatting of the data source,
    combining columns into sensible datetime time series Series

    must include "UTC_DATE" and "UTC_TIME" columns and not have missing values
    """
    dt_series = df.apply(lambda x: x['UTC_DATE'] + x["UTC_TIME"], axis=1)
    return pd.to_datetime(dt_series, format="%Y%m%d%H%M")


if __name__ == "__main__":
    weather_df = get_data()
    # convert columns to float, replace -9999 with np.nan
    weather_df["T_HR_AVG"] = weather_df["T_HR_AVG"].astype(float)
    weather_df.loc[weather_df["T_HR_AVG"] == -9999.0, "T_HR_AVG"] = np.nan

    # need to get these before converting to int
    timestamps = get_timestamps_from_concat(weather_df)

    # convert hour of day to int
    weather_df["UTC_TIME"] = weather_df["UTC_TIME"].astype(int) / 100
    weather_df["UTC_TIME"] = weather_df["UTC_TIME"].astype(int)

    daily_temp_ranges = get_daily_temp_ranges(weather_df["T_HR_AVG"])

    # compute rolling average of temps
    rolling_avg_temps = get_rolling_average(weather_df["T_HR_AVG"].array, samples=24)

    BOULDER_LATITUDE = weather_df["LATITUDE"].astype(float).values[0]
    BOULDER_LONGITUDE = weather_df["LONGITUDE"].astype(float).values[0]

    solar_position_df = calculate_solar_position_series(
        timestamps=timestamps, latitude=BOULDER_LATITUDE, longitude=BOULDER_LONGITUDE
    )
    weather_df["clearsky_radiation"] = solar_position_df["clearsky_radiation"].copy()
    weather_df["timestamps"] = solar_position_df["timestamp"].copy()

    # solar radiation measurements
    SOLARAD_BAD_FLAG = -99999
    weather_df["SOLARAD"] = weather_df["SOLARAD"].astype(float)
    weather_df.loc[weather_df["SOLARAD"] == SOLARAD_BAD_FLAG, "SOLARAD"] = np.nan

    plt.style.use("ggplot")
    _, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(weather_df["timestamps"], weather_df["SOLARAD"], label="avg_solar_radiation")
    ax.plot(
        weather_df["timestamps"] + pd.Timedelta(hours=7),
        weather_df["clearsky_radiation"],
        label="predicted clearsky radiation",
    )

    ax2 = axes[1]
    ax2.plot(
        weather_df["timestamps"],
        weather_df["T_HR_AVG"],
        color="#1C51FE",
        linestyle="--",
        linewidth=4,
        label="average temps",
    )

    ax.set_xlabel("hours elapsed")
    ax.set_ylabel("average solar radiation [W / m^2]")
    ax2.set_ylabel("average temperature [C]")
    ax.set_title("hourly values for avg solarad")
    ax.legend()

    plt.tight_layout()
    plt.show()
