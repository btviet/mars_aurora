from datetime import datetime


obs_times = [
        datetime.strptime("2023-05-05 18:35:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-05-10 22:56:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-09-01 21:04:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2024-03-18 06:45:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-18 00:33:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-20 01:15:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-06-17 18:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-08-04 00:21:00", "%Y-%m-%d %H:%M:%S"),
] #  Actual observation times 


obs_closest_times = [
        datetime.strptime("2023-05-05 19:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-05-10 23:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-09-01 21:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2024-03-18 07:00:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-18 01:00:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-20 01:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-06-17 18:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-08-04 00:00:00", "%Y-%m-%d %H:%M:%S"),
] # The time stamps in MAVEN/SEP data closest to the observations

