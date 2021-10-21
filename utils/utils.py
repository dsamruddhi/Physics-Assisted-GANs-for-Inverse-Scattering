import random
import io
import matplotlib.pyplot as plt


class PlotUtils:

    def check_data(train_input, train_output):

        for i in random.sample(range(0, train_input.shape[0]), 5):
            print(i)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

            original = ax1.imshow(train_output[i, :, :], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Original scatterer")

            guess_real = ax2.imshow(train_input[i, :, :, 0], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
            fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
            ax2.title.set_text("Initial guess: real")

            guess_imag = ax3.imshow(train_input[i, :, :, 1], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
            fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Initial guess: imaginary")

            plt.show()

    @staticmethod
    def plot_results(gt, chi_real, chi_imag, output, mode="show"):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)

        original = ax1.imshow(gt, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb1 = fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        ax1.title.set_text(f"Original scatterer")
        ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_real = ax2.imshow(chi_real, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax2.title.set_text("Initial guess: real")
        ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_imag = ax3.imshow(chi_imag, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
        cb3.ax.tick_params(labelsize=12)
        ax3.title.set_text("Initial guess: imaginary")
        ax3.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        reconstruction = ax4.imshow(output, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb4 = fig.colorbar(reconstruction, ax=ax4, fraction=0.046, pad=0.04)
        cb4.ax.tick_params(labelsize=12)
        ax4.title.set_text("Reconstructed")
        ax4.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax3.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax4.get_xticklabels(), fontsize=12, horizontalalignment="left")

        plt.setp(ax1.get_yticklabels(), fontsize=12)
        plt.setp(ax2.get_yticklabels(), fontsize=12)
        plt.setp(ax3.get_yticklabels(), fontsize=12)
        plt.setp(ax4.get_yticklabels(), fontsize=12)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

        if mode == "show":
            plt.show()
        elif mode == "save":
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf
        else:
            raise ValueError("Invalid mode for plot results function")
