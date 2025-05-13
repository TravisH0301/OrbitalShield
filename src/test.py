from collections import defaultdict
from datetime import(
    datetime,
    timedelta,
    timezone,
)
import getpass
import json
import multiprocessing as mp
from typing import(
    Dict,
    List,
    Tuple,
)

import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.spatial import Voronoi
from sgp4.api import(
    jday,
    Satrec,
    SatrecArray,
)
from spacetrack import SpaceTrackClient


def search_conjunction_pair(
        future_ts_li: list,
        sat_orbit_dict: Dict[str, List],
        sat_ids: List[str],
        dist_threshold: int,
    ) -> Dict[datetime, float]:
        """This function searches for conjunction satellite pairs
        at the given future timestamps with miss distance under threshold.

        Args:
            future_ts_li: List of future timestamps.
            sat_orbit_dict: Dictionary of satellite ids with their future orbits.
            sat_ids: List of satellite IDs.
            dist_threshold: Threshold of miss distance in km.

        Returns:
            Dictionary of conjunction pairs with a list of values
            of future timestamp and miss distance.
        """
        pair_ts_dist_dict_mp = defaultdict(list)
        for i, ts in enumerate(future_ts_li):
            # Collect satellite positions at iterating timestamp
            positions = []
            for sat_id, orbits in sat_orbit_dict.items():
                positions.append(orbits[i])

            # Map Voronoi diagram to identify neighbouring satellites
            vor = Voronoi(positions)
            for idx1, idx2 in vor.ridge_points:
                sat1 = sat_ids[idx1]
                sat2 = sat_ids[idx2]

                # Collect pairs with Euclidean distance < threshold
                dist = norm(positions[idx1] - positions[idx2])
                if dist < dist_threshold:
                    pair = tuple(sorted((sat1, sat2)))
                    pair_ts_dist_dict_mp[pair].append((ts, dist))

        return pair_ts_dist_dict_mp


if __name__ == "__main__":
    # Define Space-Track credential
    ST_USER = input("Username:")
    ST_PWD = getpass.getpass("Password:")

    # Fetch latest satellite records
    with SpaceTrackClient(ST_USER, ST_PWD) as st:
        resp = st.gp(
            epoch=">now-1",
            format="json",
        )

    """Use metadata from JSON to create satellite dim table"""
    # Convert JSON records to dataframe
    df = pd.DataFrame(json.loads(resp))

    # Define array of satellite objects based on TLE data
    sat_arry = SatrecArray([
        Satrec.twoline2rv(t1, t2) for t1, t2 in df[["TLE_LINE1", "TLE_LINE2"]].to_numpy()
    ])

    # Define future timestamps to propagate orbits
    diff_min = 5
    total_min = 60 * 24
    curr_ts = datetime.now(timezone.utc)
    future_ts_li = []
    for i in range(diff_min, total_min + diff_min, diff_min):
        ts = curr_ts + timedelta(minutes=i)
        future_ts_li.append(ts)

    # Convert timestamps to Julian dates
    jd_arry = fr_arry = np.empty(0)
    for ts in future_ts_li:
        jd, fr = jday(
            ts.year,
            ts.month,
            ts.day,
            ts.hour,
            ts.minute,
            ts.second + ts.microsecond / 1e6
        )
        jd_arry = np.append(jd_arry, jd)
        fr_arry = np.append(fr_arry, fr)

    # Propagate orbits using future dates
    e, r, v = sat_arry.sgp4(jd_arry, fr_arry)  # error, position[x, y, z], velocity

    # Define dict for satellite IDs and their propagated orbits
    """
    sat_orbit_dict = {
        sat_id_1: [
            [x_t1, y_t1, z_t1],
            ...
            [x_tn, y_tn, z_tn],
        ],
        ...
        sat_id_n:...
    }
    """
    sat_orbit_dict = dict(zip(df["NORAD_CAT_ID"], r))

    # Loop through future timestamps to find conjunction candidates
    # at TCA (Time of Closest Approach) with miss distance under threshold
    dist_threshold = 5  # km
    sat_ids = list(sat_orbit_dict.keys())
    pair_ts_dist_dict = defaultdict(list)

    

    # Determine conjunction pairs via multiprocessing
    with mp.Pool(mp.cpu_count()) as pool:
        cpu_count = mp.cpu_count()
        k, m = divmod(len(future_ts_li), cpu_count)
        future_ts_li_mp = [
            future_ts_li[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(cpu_count)
        ]
        args = [
            (chunk, sat_orbit_dict, sat_ids, dist_threshold)
            for chunk in future_ts_li_mp
        ]
        results = pool.starmap(search_conjunction_pair, args)

    # Merge results from multiprocessing
    for dict_ in results:
        for pair, ts_dist_li in dict_.items():
            pair_ts_dist_dict[pair].extend(ts_dist_li)

    # Determine miss distance at TCA for conjunction candidates
    conjunction_pairs = []
    for pair, ts_dist_li in pair_ts_dist_dict.items():
        if not ts_dist_li:
            continue

        tca, dist = min(ts_dist_li, key=lambda x: x[1])
        conjunction_pairs.append({
            "Satellite 1": pair[0],
            "Satellite 2": pair[1],
            "TCA": tca,
            "Miss Distance (km)": dist,
        })

    print(conjunction_pairs)