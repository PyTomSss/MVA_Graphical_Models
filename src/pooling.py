import torch

def pooling_op(x, readout, batch=None, device=None):
  if len(x.size()) == 3:
    if readout == 'max':
      return torch.max(x, dim=1)[0].squeeze() # max readout
    elif readout == 'avg':
      return torch.mean(x, dim=1).squeeze() # avg readout
    elif readout == 'sum':
      return torch.sum(x, dim=1).squeeze() # sum readout
  elif len(x.size()) == 2:

    if batch is None:
        # single graph, pas de batching
        if readout == 'max':
            return torch.max(x, dim=0)[0].squeeze()
        elif readout == 'mean':
            return torch.mean(x, dim=0).squeeze()
        elif readout == 'sum':
            return torch.sum(x, dim=0).squeeze()
        else:
            raise ValueError(f"Unknown readout type: {readout}")

    batch = batch.cpu().tolist()
    readouts = []
    max_batch = max(batch)
    
    temp_b = 0
    last = 0
    for i, b in enumerate(batch):
      if b != temp_b:
        sub_x = x[last:i]
        if readout == 'max':
          readouts.append(torch.max(sub_x, dim=0)[0].squeeze()) # max readout
        elif readout == 'avg':
          readouts.append(torch.mean(sub_x, dim=0).squeeze()) # avg readout
        elif readout == 'sum':
          readouts.append(torch.sum(sub_x, dim=0).squeeze()) # sum readout
                  
        last = i
        temp_b = b
      elif b == max_batch:
        sub_x = x[last:len(batch)]
        if readout == 'max':
          readouts.append(torch.max(sub_x, dim=0)[0].squeeze()) # max readout
        elif readout == 'avg':
          readouts.append(torch.mean(sub_x, dim=0).squeeze()) # avg readout
        elif readout == 'sum':
          readouts.append(torch.sum(sub_x, dim=0).squeeze()) # sum readout
                  
        break
        
    readouts = torch.cat(readouts, dim=0)
    return readouts