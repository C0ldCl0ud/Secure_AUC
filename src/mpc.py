import crypten
import torch


def encrypt(vector, parties=2):

    # transform
    values = vector.iloc[:, 0].tolist()

    @crypten.mpc.run_multiprocess(world_size=parties)
    def encr(values):
        x = torch.tensor(values)
        x_enc = crypten.cryptensor(x)
        return x_enc

    res = encr(values)

    return res