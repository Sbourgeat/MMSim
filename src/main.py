from topology import persistence_homology
from data import import_data



def main():
    # import data

    data = import_data.import_price_historical("AAPL")

    ts = data["Close"].values

    ts_embedded = persistence_homology.embed_time_series(ts)



    if ts_embedded.shape[1] > 3:
        ts_embedded = persistence_homology.reduce_dimension(ts_embedded)

    # create plot 
    persistence_homology.plot_pcd(ts_embedded, "embedded_ts_AAPL")


    # Compute VR filtration

    persistence_diagram = persistence_homology.vietoris_rips_transform(ts_embedded, symbol="AAPL")

    persistence_entropy = persistence_homology.persistence_entropy(persistence_diagram)

    print(f"The computed persistence diagram is:\nH0{persistence_entropy[0][0]}\nH1{persistence_entropy[0][1]}\nH2{persistence_entropy[0][2]}")


if __name__ == "__main__":
    main()
