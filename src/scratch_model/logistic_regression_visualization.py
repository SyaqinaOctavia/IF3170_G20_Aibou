import matplotlib.pyplot as plt # gak perlu di deepnote 
from matplotlib.animation import FuncAnimation
import numpy as np # gak perlu di deepnote 

from .logistic_regression import LogisticRegression # gak perlu di deepnote 


def create_loss_contour_animation(
    X,
    y,
    feature_name=None,
    max_iter=100,
    learning_rate=0.1,
    sample_size=1000,
    random_state=67,
    grid_size=55,
    frame_count=60,
):

    if not isinstance(frame_count, (int, np.integer)) or frame_count < 2:
        raise ValueError("frame_count harus berupa bilangan bulat minimal 2")

    if hasattr(X, "columns"):
        if feature_name is None:
            if len(X.columns) != 1:
                raise ValueError("Pilih satu fitur melalui feature_name untuk visualisasi dua parameter")
            feature_name = X.columns[0]
        if feature_name not in X.columns:
            raise ValueError(f"Fitur {feature_name!r} tidak ditemukan")
        values = X[feature_name].to_numpy(dtype=float)
    else:
        values = np.asarray(X, dtype=float)
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError("Visualisasi kontur theta0-theta1 membutuhkan tepat satu fitur")
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("X harus berupa satu fitur")

    labels = np.asarray(y).reshape(-1)
    if len(values) != len(labels):
        raise ValueError("X dan y harus memiliki jumlah baris yang sama")

    valid = np.isfinite(values)
    if np.issubdtype(labels.dtype, np.number):
        valid &= np.isfinite(labels)
    values, labels = values[valid], labels[valid]
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("Visualisasi ini memerlukan target dengan tepat dua kelas")
    if len(values) < 2:
        raise ValueError("Data valid terlalu sedikit untuk membuat visualisasi")

    rng = np.random.default_rng(random_state)
    if sample_size is not None and len(values) > sample_size:
        indices = rng.choice(len(values), size=sample_size, replace=False)
        values, labels = values[indices], labels[indices]

    targets = (labels == classes[1]).astype(float)
    mean = values.mean()
    scale = values.std()
    if scale == 0:
        raise ValueError("Fitur harus memiliki variasi nilai")
    scaled_values = ((values - mean) / scale).reshape(-1, 1)

    model = LogisticRegression(
        penalty=None,
        max_iter=max_iter,
        learning_rate=learning_rate,
        batch_size=None,
        random_state=random_state,
        record_history=True,
        tol=0.0,
    )
    model.fit(scaled_values, targets)

    # Urutan parameter model adalah [bobot fitur, bias].
    theta_path = model.theta_history_[:, 0, :]
    theta0_path = theta_path[:, 1]
    theta1_path = theta_path[:, 0]
    frame_indices = np.unique(
        np.linspace(
            0,
            len(theta0_path) - 1,
            min(frame_count, len(theta0_path)),
            dtype=int,
        )
    )

    pad0 = max(float(np.ptp(theta0_path)) * 0.2, 0.5)
    pad1 = max(float(np.ptp(theta1_path)) * 0.2, 0.5)
    grid0 = np.linspace(theta0_path.min() - pad0, theta0_path.max() + pad0, grid_size)
    grid1 = np.linspace(theta1_path.min() - pad1, theta1_path.max() + pad1, grid_size)
    T0, T1 = np.meshgrid(grid0, grid1)
    flat0, flat1 = T0.ravel(), T1.ravel()
    flat_losses = np.empty_like(flat0)
    x_values = scaled_values[:, 0]

    for start in range(0, len(flat0), 64):
        end = min(start + 64, len(flat0))
        scores = flat0[start:end, None] + flat1[start:end, None] * x_values[None, :]
        batch_losses = np.logaddexp(0.0, scores) - targets[None, :] * scores
        flat_losses[start:end] = batch_losses.mean(axis=1)
    losses = flat_losses.reshape(grid_size, grid_size)

    # Kontur parameter di atas dan riwayat loss training di bawah.
    fig = plt.figure(figsize=(9, 7), dpi=75)
    layout = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.15), hspace=0.38)
    ax_contour = fig.add_subplot(layout[0])
    ax_loss = fig.add_subplot(layout[1])
    fig.suptitle(
        "Jejak Pelatihan Logistic Regression",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="semibold",
    )
    fig.text(
        0.08,
        0.925,
        f"Fitur: {feature_name or 'satu fitur'}  |  Sampel: {len(targets)}  |  "
        f"Learning rate: {learning_rate:g}  |  Tanpa regularisasi",
        fontsize=9,
        color="#52606d",
    )
    status_text = fig.text(
        0.08,
        0.895,
        "",
        ha="left",
        fontsize=9,
        color="#333333",
    )

    contours = ax_contour.contour(
        T0,
        T1,
        losses,
        levels=14,
        colors="#777777",
        linewidths=0.8,
    )
    ax_contour.clabel(contours, inline=True, fontsize=8, fmt="%.3f")
    ax_contour.set_xlabel(r"$\theta_0$ (bias)")
    ax_contour.set_ylabel(r"$\theta_1$ (bobot fitur)")
    ax_contour.set_title("Lintasan parameter pada permukaan loss", loc="left", fontsize=11)
    ax_contour.grid(alpha=0.12)
    path_line, = ax_contour.plot([], [], color="#222222", linewidth=1.8)
    current_point, = ax_contour.plot(
        [], [], marker="o", markerfacecolor="white", markeredgecolor="#222222", markersize=7
    )

    loss_values = np.concatenate(([np.log(2.0)], np.asarray(model.loss_history_)))
    epochs = np.arange(len(loss_values))
    ax_loss.set_title("Log-loss training (binary cross-entropy)", loc="left", fontsize=11)
    ax_loss.set_xlabel("Iterasi")
    ax_loss.set_ylabel("Rata-rata log-loss")
    ax_loss.set_xlim(0, max(1, len(loss_values) - 1))
    low, high = float(loss_values.min()), float(loss_values.max())
    loss_pad = max((high - low) * 0.08, 0.01)
    ax_loss.set_ylim(low - loss_pad, high + loss_pad)
    ax_loss.grid(alpha=0.25)
    loss_line, = ax_loss.plot([], [], color="#444444", linewidth=1.8)
    loss_point, = ax_loss.plot([], [], marker="o", color="#222222", markersize=5)

    def update(frame):
        path_line.set_data(theta0_path[: frame + 1], theta1_path[: frame + 1])
        current_point.set_data([theta0_path[frame]], [theta1_path[frame]])
        loss_line.set_data(epochs[: frame + 1], loss_values[: frame + 1])
        loss_point.set_data([epochs[frame]], [loss_values[frame]])
        current_loss = loss_values[frame]
        status_text.set_text(
            f"Iterasi {frame}/{len(theta0_path) - 1}  |  "
            f"Bias (intersep): {theta0_path[frame]:.3f}  |  "
            f"Bobot fitur {feature_name or 'fitur'}: {theta1_path[frame]:.3f}  |  "
            f"Log-loss: {current_loss:.4f}"
        )
        return path_line, current_point, loss_line, loss_point, status_text

    fig.subplots_adjust(left=0.1, right=0.96, top=0.84, bottom=0.1)
    animation = FuncAnimation(
        fig, update, frames=frame_indices, interval=100, blit=False
    )
    return animation

# Tunjukin visualiasi di deepnote:
# from IPython.display import HTML, display
# import matplotlib.pyplot as plt

# anim = create_loss_contour_animation(
#     X_viz,
#     y_train,
#     feature_name="days_before_arrival",
#     max_iter=1000,
#     frame_count=60,
# )

# video_html = anim.to_jshtml(fps=12, default_mode="once")
# plt.close(anim._fig)
# display(HTML(video_html))