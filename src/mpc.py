import crypten
import torch


def encrypt(vector, parties):

    # transform
    values = vector.iloc[:, 0].tolist()

    @crypten.mpc.run_multiprocess(world_size=parties)
    def encr(values):
        x = torch.tensor(vector)
        x_enc = crypten.cryptensor(x)
        return x_enc

    res = encr(vector)

    return res