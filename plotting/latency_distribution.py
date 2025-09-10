import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats


# https://en.wikipedia.org/wiki/Log-normal_distribution
def plot_lognormal_latency(median_ms, sigma, output_file="latency_distribution.png"):
    """
    Plot the lognormal latency distribution used by the service.

    Args:
        median_ms: Median latency in milliseconds (note: not the mean)
        sigma:    Lognormal sigma parameter (std dev of ln(latency))
        output_file: Output file name for the plot
    """
    # For a lognormal: median = exp(mu)  =>  mu = ln(median)
    mu = np.log(median_ms)

    # x values (latency range in milliseconds)
    x_ms = np.logspace(np.log10(median_ms * 0.1), np.log10(median_ms * 10), 1000)

    # PDF and CDF in ms units
    pdf = stats.lognorm.pdf(x_ms, s=sigma, scale=np.exp(mu))  # units: 1/ms
    cdf = stats.lognorm.cdf(x_ms, s=sigma, scale=np.exp(mu))  # unitless

    # subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot PDF
    ax1.plot(
        x_ms,
        pdf,
        "b-",
        linewidth=2,
        label=f"PDF (median={median_ms} ms, σ={sigma})",
    )
    ax1.axvline(
        median_ms,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Median ({median_ms} ms)",
    )
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Lognormal Latency Distribution - Probability Density Function")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, median_ms * 5)

    # Plot CDF
    ax2.plot(
        x_ms, cdf, "g-", linewidth=2, label=f"CDF (median={median_ms} ms, σ={sigma})"
    )
    ax2.axvline(
        median_ms,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Median ({median_ms} ms)",
    )
    ax2.axhline(0.5, color="r", linestyle="--", alpha=0.7)

    # Add percentile lines
    percentiles = [50, 90, 95, 99]
    colors = ["red", "orange", "purple", "brown"]
    for p, color in zip(percentiles, colors):
        p_value_ms = stats.lognorm.ppf(p / 100, s=sigma, scale=np.exp(mu))
        if p_value_ms <= median_ms * 5:
            ax2.axvline(
                p_value_ms,
                color=color,
                linestyle=":",
                alpha=0.7,
                label=f"p{p} ({p_value_ms:.1f} ms)",
            )

    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Lognormal Latency Distribution - Cumulative Distribution Function")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, median_ms * 5)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved latency distribution plot to {output_file}")

    # Percentiles (ms)
    p50 = stats.lognorm.ppf(0.5, s=sigma, scale=np.exp(mu))
    p90 = stats.lognorm.ppf(0.9, s=sigma, scale=np.exp(mu))
    p95 = stats.lognorm.ppf(0.95, s=sigma, scale=np.exp(mu))
    p99 = stats.lognorm.ppf(0.99, s=sigma, scale=np.exp(mu))

    print("\nLatency Percentiles:")
    print(f"p50 (median): {p50:.2f} ms")
    print(f"p90: {p90:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    print(f"p99: {p99:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot lognormal latency distribution")
    parser.add_argument(
        "--median",
        type=float,
        default=20.0,
        help="Median latency in milliseconds (default: 20.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Lognormal sigma parameter (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="latency_distribution.png",
        help="Output plot filename",
    )

    args = parser.parse_args()

    plot_lognormal_latency(args.median, args.sigma, args.output)
