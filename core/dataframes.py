import polars as pl
import os

data_dir = "./data"
cond_fname = "MT_COND.csv"
tree_fname = "MT_TREE.csv"
plot_fname = "MT_PLOT.csv"

cond_selected_columns = ["CN", "PLT_CN", "SLOPE", "ASPECT", "BALIVE"]
tree_selected_columns = ["CN", "PLT_CN", "DIA", "DIAHTCD", "HT", "ACTUALHT", "CR"]
plot_selected_columns = ["CN", "PLOT", "ELEV"]

def main():
    print("Creating data frames")

    cond = pl.read_csv(os.path.join(data_dir, cond_fname), columns=cond_selected_columns)
    tree = pl.read_csv(os.path.join(data_dir, tree_fname), columns=tree_selected_columns)
    plot = pl.read_csv(os.path.join(data_dir, plot_fname), columns=plot_selected_columns)

    print(cond)
    print(plot)
    print(tree)

if __name__ == "__main__":
   main()
