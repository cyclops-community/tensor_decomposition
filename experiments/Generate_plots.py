import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_results(residuals, norm_mahalanobis, final_residuals, mean_residuals, std_residuals, s, rank, epsilon, alph):
   
    plt.figure(figsize=(14, 8))

    # 1. Iterations vs Residual
    plt.subplot(2, 2, 1)
    iterations = []
    iterations.append(len(residuals[0]))
    iterations.append(len(residuals[1]))
    iterations.append(len(residuals[2]))
    plt.plot(range(1, iterations[0] + 1), residuals[0], marker='.',color=lighten_color('orange', 0.55),
                    linewidth=2,label=f'Rank={rank} ALS',markersize=10)
    plt.plot(range(1, iterations[1] + 1), residuals[1], marker='>',color=lighten_color('blue', 0.55),
                    linewidth=2,label=f'Rank={rank} AMDM',markersize=10)
    plt.plot(range(1, iterations[2] + 1), residuals[2], marker='>',color=lighten_color('red', 0.55),
                    linewidth=2,label=f'Rank={rank} Hybrid',markersize=10)
    # plt.title("Iterations vs Residual")
    # plt.xlabel("Iterations")
    # plt.ylabel("Residual")
    # plt.grid(True)
    # plt.legend()

    plt.xlim(left=1,right=iterations[0])
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend( prop={'size': 15})
    #plt.title(f'Noisy tensors with s={s} and the level of noise $\epsilon={epsilon}$ and $\alpha={alph}$')
    plt.title(f'Noisy Tensors with s={s} and the Level of Noise $\\epsilon={epsilon}$ and $\\alpha={alph}$')
    plt.xlabel('iterations')
    plt.ylabel("absolute residual")
    #plt.savefig('./Change_R_abs_res_s10_new.pdf',bbox_inches='tight')
    #plt.show()

    # 2. Iterations vs Norm ALS
    plt.subplot(2, 2, 2)
    iterations = []
    iterations.append(len(norm_mahalanobis[0]))
    iterations.append(len(norm_mahalanobis[1]))
    iterations.append(len(norm_mahalanobis[2]))
    plt.plot(range(1, iterations[0] + 1), norm_mahalanobis[0], marker='.',color=lighten_color('orange', 0.55),
                    linewidth=2,label=f'Rank={rank} ALS',markersize=10)
    plt.plot(range(1, iterations[1] + 1), norm_mahalanobis[1], marker='<',color=lighten_color('blue', 0.55),
                    linewidth=2,label=f'Rank={rank} AMDM',markersize=10)
    plt.plot(range(1, iterations[2] + 1), norm_mahalanobis[2], marker='<',color=lighten_color('red', 0.55),
                    linewidth=2,label=f'Rank={rank} Hybrid',markersize=10)
    # plt.title("Iterations vs Norm M")
    # plt.xlabel("Iterations")
    # plt.ylabel("Norm ALS")
    # plt.grid(True)
    # plt.legend()
    plt.xlim(left=1,right=iterations[0])
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend( prop={'size': 15})
    #plt.title(f'Noisy tensors with s={s} and the level of noise $\epsilon={epsilon}$ and $\alpha={alph}$')
    plt.title(f'Noisy Tensors with s={s} and the Level of Noise $\\epsilon={epsilon}$ and $\\alpha={alph}$')

    plt.xlabel('iterations')
    plt.ylabel(r"$||T-[A,B,C]||_{M^{-1}}$")
    #plt.savefig('./Change_R_abs_res_s10_new.pdf',bbox_inches='tight')
    #plt.show()

    # 3. Scatter Diagram of Final Residuals
    plt.subplot(2, 2, 3)
    num_runs = len(final_residuals[0])
    plt.scatter(range(1, num_runs + 1), final_residuals[0], color='orange', label=f'Rank={rank} ALS', alpha=0.7)
    plt.scatter(range(1, num_runs + 1), final_residuals[1], color='blue', label=f'Rank={rank} AMDM', alpha=0.7)
    plt.scatter(range(1, num_runs + 1), final_residuals[2], color='red', label=f'Rank={rank} Hybrid', alpha=0.7)
    # plt.title("Scatter Diagram of Final Residuals Across Runs")
    # plt.xlabel("run index')
    # plt.ylabel("final residual")
    # plt.grid(True)
    plt.xlim(left=1,right=num_runs)
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend( prop={'size': 15})
    plt.title('Scatter Diagram of Final Residuals Across Runs')
    plt.xlabel('iterations')
    plt.ylabel('absolute residual')
    #plt.savefig('./Change_R_abs_res_s10_new.pdf',bbox_inches='tight')
   # plt.show()

    # 4. Error Bar Plot (Residuals with Standard Deviation)
    plt.subplot(2, 2, 4)
    iterations_err_bar = []
    iterations_err_bar.append(len(mean_residuals[0]))
    iterations_err_bar.append(len(mean_residuals[1]))
    iterations_err_bar.append(len(mean_residuals[2]))
    plt.errorbar(range(1, iterations_err_bar[0] + 1), mean_residuals[0], yerr=std_residuals[0], fmt='-o', capsize=5,
                     label=f"ALS: Mean Residual ± Std Dev", color='orange')
    plt.errorbar(range(1, iterations_err_bar[1] + 1), mean_residuals[1], yerr=std_residuals[1], fmt='-.', capsize=5,
                     label=f"AMDM: Mean Residual ± Std Dev", color='blue')
    plt.errorbar(range(1, iterations_err_bar[2] + 1), mean_residuals[2], yerr=std_residuals[2], fmt='-.', capsize=5,
                     label=f"Hybrid: Mean Residual ± Std Dev", color='red')
    # plt.title("Residual vs Iterations with Error Bars")
        # plt.xlabel("iterations")
        # plt.ylabel("residual")
        # plt.legend()
        # plt.grid(True)
    plt.xlim(left=1,right=iterations_err_bar[0])
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend( prop={'size': 15})
    plt.title('Residual vs Iterations with Error Bars')
    plt.xlabel('iterations')
    plt.ylabel('absolute residual')
    #plt.savefig('./Change_R_abs_res_s10_new.pdf',bbox_inches='tight')
    #plt.show()

    # Show all plots
    plt.tight_layout()
    plt.show()


# # Example integration into main function
# if __name__ == "__main__":
    
#     # Assuming these variables are computed during tensor decomposition
#     iterations = np.arange(1, num_iter + 1)  # Replace with actual iteration range
#     residuals = best_run_residual             # Replace with actual residual values
#     norm_als = best_run_norm_mahalanobis_empirical  # Replace with norm values
#     final_residuals = final_residuals         # Replace with final residual values across runs
#     std_residuals = std_residuals            # Replace with standard deviation if available

#     # Call the plotting function
#     plot_results(iterations, residuals, norm_als, final_residuals, std_residuals)
