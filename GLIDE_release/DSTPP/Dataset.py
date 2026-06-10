import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import DSTPP.Constants as Constants

class PreprocessedEventDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem[0] for elem in inst] for inst in data]
        self.time_norm = [[elem[1] for elem in inst] for inst in data]
        self.lng = [[elem[2] for elem in inst] for inst in data]
        self.lat = [[elem[3] for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx],self.time_norm[idx], self.lng[idx], self.lat[idx]


class EventData_3D(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem[0] for elem in inst] for inst in data]
        self.time_norm = [[elem[1] for elem in inst] for inst in data]
        self.lng = [[elem[2] for elem in inst] for inst in data]
        self.lat = [[elem[3] for elem in inst] for inst in data]
        self.height = [[elem[4] for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_norm[idx], self.lng[idx], self.lat[idx], self.height[idx]

class EventData_1D(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        
        self.time = [[elem[0] for elem in inst] for inst in data]
        self.time_norm = [[elem[1] for elem in inst] for inst in data]
        self.lng = [[elem[2] for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_norm[idx], self.lng[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_norm, lng, lat = list(zip(*insts))
    time = pad_time(time)
    time_norm = pad_time(time_norm)
    lat = pad_time(lat)
    lng = pad_time(lng)
    return time,time_norm, lng, lat
def preprocessed_collate_fn(batch):
    """
    Collate preprocessed graph dictionaries into a PyG batch.
    """
    data_list = []
    for item in batch:
      
        
        node_features = torch.as_tensor(item['x'])
      
       
        d = Data(
            x=node_features, 
            target_time=torch.as_tensor(item['target_time']),
            target_loc=torch.as_tensor(item['target_loc']),

            edge_index=torch.as_tensor(item['edge_index'], dtype=torch.long) if item['edge_index'] is not None else torch.empty((2,0), dtype=torch.long),
            edge_attr=torch.as_tensor(item['edge_attr']) if item['edge_attr'] is not None else torch.empty((0, 3)),

            num_nodes=item['length']
        )
        data_list.append(d)
    

    return Batch.from_data_list(data_list)

def collate_fn_3d(insts):
    """ Collate function, as required by PyTorch. """

    time, time_norm, lng, lat, height = list(zip(*insts))
    time = pad_time(time)
    time_norm = pad_time(time_norm)
    lat = pad_time(lat)
    lng = pad_time(lng)
    height = pad_time(height)
    return time, time_norm, lng, lat, height

def collate_fn_1d(insts):
    """ Collate function, as required by PyTorch. """

    time, time_norm, lng = list(zip(*insts))
    time = pad_time(time)
    time_norm = pad_time(time_norm)
    lng = pad_time(lng)
    return time, time_norm, lng


def get_dataloader(data, batch_size, D = 2, shuffle=True):
    """ Prepare dataloader. """

    collate = {1:collate_fn_1d, 2:collate_fn, 3:collate_fn_3d}

    if D>=2:
        ds = EventData(data) if D==2 else EventData_3D(data)
    if D==1:
        ds = EventData_1D(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=16,
        batch_size=batch_size,
        collate_fn= collate[D],
        shuffle=shuffle,
        persistent_workers=True, 
        prefetch_factor=4,
        pin_memory=True
    )
    return dl
def get_dataloader_preprocessed(data, batch_size, shuffle=True):
    """
    Create a DataLoader for preprocessed graph examples.
    """
    dataset = PreprocessedEventDataset(data)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=preprocessed_collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
