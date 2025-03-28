# ----- LOAD LIBRARIES -----
library(tm)
library(textstem)    # for lemmatization
library(factoextra)  # for cluster visualizations (fviz_nbclust, fviz_cluster)
library(cluster)     # for silhouette analysis
library(proxy)       # for cosine similarity/distance
library(ggplot2)

# ----- LOAD & CLEAN DATA (if not already done) -----
# (Assuming you have your combined_df from your CSV files with a 'content' column)
# For this demo, we assume you have already built a DTM and filtered it.
# Here we rebuild the DTM quickly for completeness:

aiart_path <- "/Users/roryoflynn/Desktop/TEXT MINING/MOD1ASSGINMENT/LDF-NKW-AiArt.csv"
art_path   <- "/Users/roryoflynn/Desktop/TEXT MINING/MOD1ASSGINMENT/LDF-NKW-Art.csv"

aiart_df <- read.csv(aiart_path, stringsAsFactors = FALSE)
art_df   <- read.csv(art_path, stringsAsFactors = FALSE)

# Drop extra index column
aiart_df <- aiart_df[,-1]
art_df   <- art_df[,-1]

# Combine data
combined_df <- rbind(aiart_df, art_df)
cat("Total documents in combined data:", nrow(combined_df), "\n")

# Create and clean corpus
corpus <- Corpus(VectorSource(combined_df$content))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))
corpus <- tm_map(corpus, stripWhitespace)

# Create Document-Term Matrix with TF-IDF weighting
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
dtm_mat <- as.matrix(dtm)
cat("Initial DTM shape:", dim(dtm_mat), "\n")

# Filter out very rare or overly common terms
term_freq <- colSums(dtm_mat > 0)
total_docs <- nrow(dtm_mat)
keep_terms <- names(term_freq)[term_freq >= 3 & term_freq <= 0.8 * total_docs]
dtm_filtered <- dtm_mat[, keep_terms]
cat("Filtered DTM shape:", dim(dtm_filtered), "\n")

# ----- DIMENSIONALITY REDUCTION VIA PCA -----
# For visualization, reduce to 2 dimensions.
pca_res <- prcomp(dtm_filtered, scale. = TRUE)
pca_data <- as.data.frame(pca_res$x[, 1:2])
colnames(pca_data)[1:2] <- c("PC1", "PC2")
# (Optional: Retain original label for later comparison)
pca_data$LABEL <- combined_df$LABEL

# ----- K-MEANS CLUSTERING: MULTIPLE k VALUES -----
# Elbow and Silhouette analyses help decide the best k.
set.seed(42)
k_range <- 2:10

# 1. Elbow method: visualize within-cluster sum-of-squares (WSS)
fviz_nbclust(pca_data[, c("PC1", "PC2")], kmeans, method = "wss", k.max = 10) +
  ggtitle("Elbow Method for K-Means Clustering")

# 2. Silhouette method: visualize average silhouette width for each k
fviz_nbclust(pca_data[, c("PC1", "PC2")], kmeans, method = "silhouette", k.max = 10) +
  ggtitle("Silhouette Analysis for K-Means Clustering")

# From these plots, assume the best k is determined as follows:
# (In your earlier Python code, silhouette peaked at k=3; adjust if needed)
best_k <- 3

# ----- FINAL K-MEANS CLUSTERING -----
km_res <- kmeans(pca_data[, c("PC1", "PC2")], centers = best_k, nstart = 25)
pca_data$kmeans_cluster <- as.factor(km_res$cluster)

# Visualize k-means clusters in PCA space
fviz_cluster(km_res, data = pca_data[, c("PC1", "PC2")],
             ellipse.type = "convex",
             geom = "point",
             main = paste("K-Means Clusters (k =", best_k, ") in PCA Space"))

# ----- HIERARCHICAL CLUSTERING (hclust) -----
# Compute cosine distance on the PCA data
cos_dist <- as.dist(1 - proxy::simil(as.matrix(pca_data[, c("PC1", "PC2")]), method = "cosine"))

# Hierarchical clustering using Ward's method
hc_res <- hclust(cos_dist, method = "ward.D2")
# Plot full dendrogram
plot(hc_res, main = "Hierarchical Clustering Dendrogram (Cosine, Ward)",
     xlab = "Documents", sub = "", cex = 0.8)
# Draw rectangles to indicate clusters (using best_k = 3)
rect.hclust(hc_res, k = best_k, border = "red")

# Optionally, determine hierarchical clusters for each document
hc_clusters <- cutree(hc_res, k = best_k)
# Compare the hierarchical clusters with k-means clusters
cat("\nContingency Table: K-Means vs. Hierarchical Clustering\n")
print(table(KMeans = pca_data$kmeans_cluster, HClust = hc_clusters))




# Convert cluster assignments to numeric if necessary:
cluster_labels <- as.numeric(as.character(pca_data$cluster))

# Now, for each cluster, produce a word cloud.
for (k in sort(unique(cluster_labels))) {
  cat("\nGenerating word cloud for Cluster", k, "...\n")
  
  # Subset the original text (from combined_df) using the cluster labels.
  docs_in_cluster_k <- combined_df$content[cluster_labels == k]
  
  # Combine all documents in this cluster into one long string.
  big_text <- paste(docs_in_cluster_k, collapse = " ")
  
  # Create a mini corpus from this combined text.
  mini_corpus <- Corpus(VectorSource(big_text))
  
  # Apply the same cleaning steps as before.
  mini_corpus <- tm_map(mini_corpus, content_transformer(tolower))
  mini_corpus <- tm_map(mini_corpus, removePunctuation)
  mini_corpus <- tm_map(mini_corpus, removeNumbers)
  mini_corpus <- tm_map(mini_corpus, removeWords, stopwords("english"))
  mini_corpus <- tm_map(mini_corpus, stripWhitespace)
  
  # Build a Term-Document Matrix.
  tdm_k <- TermDocumentMatrix(mini_corpus)
  m_k <- as.matrix(tdm_k)
  
  # Compute word frequencies.
  freq_k <- sort(rowSums(m_k), decreasing = TRUE)
  
  # Create the word cloud.
  wordcloud(names(freq_k), freq_k, max.words = 50, random.order = FALSE,
            colors = brewer.pal(8, "Dark2"),
            main = paste("Word Cloud for Cluster", k))
  
  # Optionally, you can also print the top words:
  cat("Top words in Cluster", k, ":\n")
  print(head(freq_k, 10))
}

