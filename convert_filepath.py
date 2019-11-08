import pickle

filename = 'imgs_skipped_1.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)

def convert_path(path):
    s = path.split('/')[4:]
    s = ['', 'home', 'wilson', 'causal-infogan', 'data'] + s
    s = '/'.join(s)
    return s

new_data = list()
for d in data:
    (s1, i1), (s2, i2) = d
    s1, s2 = convert_path(s1), convert_path(s2)
    new_data.append(((s1, i1), (s2, i2)))

with open(filename, 'wb') as f:
    pickle.dump(new_data, f)
