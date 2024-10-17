class corresponding(object):
    def __init__(self):
        self.seq_num_to_time = {
            12: "2019-10-24-18-08-33",
            13: "2019-10-24-18-24-40",
            14: "2019-10-24-18-21-18",
            15: "2019-10-24-18-16-24",
            16: "2019-10-24-18-13-21",
            17: "2019-10-24-18-11-06",
            18: "2019-10-24-18-05-22",
            19: "2019-10-24-18-03-57",
            20: "2019-10-24-17-58-56",
            21: "2019-10-24-17-57-46",
            22: "2019-10-24-17-56-33",
            23: "2019-10-24-18-18-47",
            27: "2019-10-24-17-55-21",
            28: "2019-10-24-17-53-58",
            29: "2019-10-24-18-06-56",
            30: "2019-10-24-17-51-58",
        }
        self.time_to_seq_num = {v: k for k, v in self.seq_num_to_time.items()}

    def time_list_to_seq(self, time_list):
        return [self.time_to_seq_num[time] for time in time_list]
    
    def seq_list_to_time(self, seq_list):
        return [self.seq_num_to_time[seq] for seq in seq_list]