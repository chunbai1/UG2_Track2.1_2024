import torch
import string

def get_str_list(output, target, dataset=None):
  assert output.dim() == 2 and target.dim() == 2

  end_label = dataset.char2id[dataset.EOS]
  unknown_label = dataset.char2id[dataset.UNKNOWN]
  num_samples, max_len_labels = output.size()
  num_classes = len(dataset.char2id.keys())
  assert num_samples == target.size(0) and max_len_labels == target.size(1)
  output = to_numpy(output)
  target = to_numpy(target)

  # list of char list
  pred_list, targ_list = [], []
  for i in range(num_samples):
    pred_list_i = []
    for j in range(max_len_labels):
      if output[i, j] != end_label:
        if output[i, j] != unknown_label:
          pred_list_i.append(dataset.id2char[output[i, j]])
      else:
        break
    pred_list.append(pred_list_i)

  for i in range(num_samples):
    targ_list_i = []
    for j in range(max_len_labels):
      if target[i, j] != end_label:
        if target[i, j] != unknown_label:
          targ_list_i.append(dataset.id2char[target[i, j]])
      else:
        break
    targ_list.append(targ_list_i)

  if True:
    pred_list = [_normalize_text(pred) for pred in pred_list]
    targ_list = [_normalize_text(targ) for targ in targ_list]
  else:
    pred_list = [''.join(pred) for pred in pred_list]
    targ_list = [''.join(targ) for targ in targ_list]

  return pred_list, targ_list

class DataInfo(object):
  """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """
  def __init__(self, voc_type):
    super(DataInfo, self).__init__()
    self.voc_type = voc_type

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)

def to_numpy(tensor):
  if torch.is_tensor(tensor):
    return tensor.cpu().numpy()
  elif type(tensor).__module__ != 'numpy':
    raise ValueError("Cannot convert {} to numpy array"
                     .format(type(tensor)))
  return tensor

def to_torch(ndarray):
  if type(ndarray).__module__ == 'numpy':
    return torch.from_numpy(ndarray)
  elif not torch.is_tensor(ndarray):
    raise ValueError("Cannot convert {} to torch tensor"
                     .format(type(ndarray)))
  return ndarray

def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
  '''
  voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
  '''
  voc = None
  types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
  if voc_type == 'LOWERCASE':
    voc = list(string.digits + string.ascii_lowercase)
  elif voc_type == 'ALLCASES':
    voc = list(string.digits + string.ascii_letters)
  elif voc_type == 'ALLCASES_SYMBOLS':
    voc = list(string.printable[:-6])
  else:
    raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

  # update the voc with specifical chars
  voc.append(EOS)
  voc.append(PADDING)
  voc.append(UNKNOWN)

  return voc

def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text.lower()

## param voc: the list of vocabulary
def char2id(voc):
  return dict(zip(voc, range(len(voc))))

def id2char(voc):
  return dict(zip(range(len(voc)), voc))

def labels2strs(labels, id2char, char2id):
  # labels: batch_size x len_seq
  if labels.ndimension() == 1:
    labels = labels.unsqueeze(0)
  assert labels.dim() == 2
  labels = to_numpy(labels)
  strings = []
  batch_size = labels.shape[0]

  for i in range(batch_size):
    label = labels[i]
    string = []
    for l in label:
      if l == char2id['EOS']:
        break
      else:
        string.append(id2char[l])
    string = ''.join(string)
    strings.append(string)

  return strings