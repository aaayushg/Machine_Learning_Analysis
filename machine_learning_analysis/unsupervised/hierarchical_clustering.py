#!/usr/bin/env python3

from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from machine_learning_analysis.common import (
    build_parser,
    configure_plotting,
    dataset_stem,
    ensure_output_dir,
    load_csv,
    print_dataframe_summary,
    require_columns,
    save_figure,
    write_json_report,
)


def parse_args():
    parser = build_parser("Run k-means and hierarchical clustering on two selected features.")
    parser.add_argument("--features", nargs=2, default=["X1", "X2"], help="Exactly two columns to cluster and plot.")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters.")
    parser.add_argument(
        "--hierarchical-linkage",
        choices=["ward", "average", "complete", "single"],
        default="ward",
        help="Linkage criterion for agglomerative clustering.",
    )
    parser.add_argument(
        "--dendrogram-linkage",
        choices=["ward", "average", "complete", "single"],
        default="average",
        help="Linkage criterion for the dendrogram rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()
    data = load_csv(args.data)
    print_dataframe_summary(data)
    require_columns(data, args.features)
    x_data = data[args.features]

    kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.random_state)
    kmeans_labels = kmeans.fit_predict(x_data)
    kmeans_silhouette = float(silhouette_score(x_data, kmeans_labels, metric="sqeuclidean"))

    hierarchy = AgglomerativeClustering(linkage=args.hierarchical_linkage, n_clusters=args.clusters)
    hierarchy_labels = hierarchy.fit_predict(x_data)
    hierarchy_silhouette = float(silhouette_score(x_data, hierarchy_labels, metric="sqeuclidean"))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x_data.iloc[:, 0], x_data.iloc[:, 1], c=kmeans_labels, cmap="tab10")
    axes[0].set_title(f"KMeans (s={kmeans_silhouette:.2f})")
    axes[0].set_xlabel(args.features[0])
    axes[0].set_ylabel(args.features[1])
    axes[1].scatter(x_data.iloc[:, 0], x_data.iloc[:, 1], c=hierarchy_labels, cmap="tab10")
    axes[1].set_title(f"Hierarchical (s={hierarchy_silhouette:.2f})")
    axes[1].set_xlabel(args.features[0])
    axes[1].set_ylabel(args.features[1])

    output_dir = ensure_output_dir(args.output_dir)
    clustering_plot = output_dir / f"{dataset_stem(args.data)}_clustering.png"
    save_figure(figure, clustering_plot)

    dendrogram_figure = plt.figure(figsize=(12, 8))
    linkage_matrix = linkage(x_data, method=args.dendrogram_linkage)
    dendrogram(linkage_matrix, labels=data.index.astype(str).tolist())
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Row index")
    plt.ylabel("Distance")
    dendrogram_plot = output_dir / f"{dataset_stem(args.data)}_dendrogram.png"
    save_figure(dendrogram_figure, dendrogram_plot)

    report = {
        "features": args.features,
        "clusters": args.clusters,
        "hierarchical_linkage": args.hierarchical_linkage,
        "dendrogram_linkage": args.dendrogram_linkage,
        "kmeans_silhouette": round(kmeans_silhouette, 4),
        "hierarchical_silhouette": round(hierarchy_silhouette, 4),
    }
    report_path = output_dir / f"{dataset_stem(args.data)}_clustering.json"
    write_json_report(report, report_path)

    print(f"KMeans silhouette: {report['kmeans_silhouette']}")
    print(f"Hierarchical silhouette: {report['hierarchical_silhouette']}")
    print(f"Cluster plot written to: {clustering_plot}")
    print(f"Dendrogram written to: {dendrogram_plot}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()