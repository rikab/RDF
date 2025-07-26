import numpy as np
import torch
import pickle

def get_pdf_toy(alpha, example, tt, order, device):
    alpha = torch.as_tensor(alpha, device=device)[..., None]
    if example == "exponential":
        if order == -1:
            y = alpha * torch.exp(-alpha * tt)
        elif order == 1:
            y = alpha.expand_as(alpha * tt) * 0 + alpha
        elif order == 2:
            y = alpha * (1 - alpha * tt)
    elif example == "angularity":
        if order == -1:
            y = alpha * tt * torch.exp(-alpha * tt**2 / 2)
        elif order == 1:
            y = alpha * tt
        elif order == 2:
            y = alpha * tt * (1 - alpha * tt**2 / 2)
    elif example == "LO_thrust":

        t_adjusted = tt  # - torch.log(torch.tensor([2.0]))
        if order == 1:

            y = torch.exp(-2 * t_adjusted) * (
                -3 * (-3 + torch.exp(t_adjusted)) * (1 + torch.exp(t_adjusted))
                + (
                    2
                    * torch.exp(t_adjusted)
                    * (
                        3
                        + torch.exp(t_adjusted)
                        * (-3 + 2 * torch.exp(t_adjusted))
                    )
                    * torch.log(-2 + torch.exp(t_adjusted))
                )
                / (-1 + torch.exp(t_adjusted))
            )

            y = (alpha * 2 / 3 / np.pi) * y
            y = torch.nan_to_num(y)
            y[y < 0] = 0

        elif order == -1:

            y = (
                alpha
                * (4 / 3)
                * (1.0 / (2 * np.pi))
                * (4 * t_adjusted - 3)
                * torch.exp(
                    -alpha
                    * (4 / 3)
                    * (1.0 / (2 * np.pi))
                    * (2 * torch.pow(t_adjusted, 2) - 3 * t_adjusted)
                )
            )
            y[y < 0] = 0

    else:
        raise ValueError("bad example/order")
    # y[y < 0] = 0
    return y.squeeze(0)



def read_in_data(file_indices, example, t_bins, device):
    from sklearn.preprocessing import MinMaxScaler

    data_dict = {}
    bin_width = t_bins.detach().cpu().numpy()[1] - t_bins.detach().cpu().numpy()[0]

    for i in file_indices:
        with open(f"event_records_LO_{i}.pkl", "rb") as ifile:
            loc_data_dict = pickle.load(ifile)
            for alpha in loc_data_dict.keys():
                loc_data = loc_data_dict[alpha][example]
                if example == "thrust":
                    loc_data = [2.0 * (1.0 - x) for x in loc_data]

                # scaler = MinMaxScaler()
                # loc_data = scaler.fit_transform(np.array(loc_data).reshape(-1,1))

                # convert to t-space
                loc_data = [np.log(1.0 / (x + 1e-12)) for x in loc_data]

                y, _ = np.histogram(
                    loc_data,
                    weights=loc_data_dict[alpha]["weight"],
                    bins=t_bins.detach().cpu().numpy(),
                    density=False,
                )
                y /= bin_width
                y = torch.tensor(y, device=device).reshape(-1, 1)
                data_dict[alpha] = y.squeeze(0).float()

    return data_dict
