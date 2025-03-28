############################################################
## Association Rule Mining (ARM) on AI/Art Text Data
## Author: [Your Name]
## Date: [Today's Date]
############################################################

# ----- 1. Load Required Libraries -----
library(tm)           # Text mining functions
library(SnowballC)    # For stemming (if desired)
library(arules)       # For ARM
library(arulesViz)    # For ARM visualizations
library(RColorBrewer) # For color palettes

# ----- 2. Load and Combine Data -----
aiart_path <- "/Users/roryoflynn/Desktop/TEXT MINING/MOD1ASSGINMENT/LDF-NKW-AiArt.csv"
art_path   <- "/Users/roryoflynn/Desktop/TEXT MINING/MOD1ASSGINMENT/LDF-NKW-Art.csv"

aiart_df <- read.csv(aiart_path, stringsAsFactors = FALSE)
art_df   <- read.csv(art_path, stringsAsFactors = FALSE)

# Remove extra index column if present (assumed first column)
aiart_df <- aiart_df[,-1]
art_df   <- art_df[,-1]

# Combine the dataframes
combined_df <- rbind(aiart_df, art_df)
cat("Total documents in combined data:", nrow(combined_df), "\n")
head(combined_df, 3)

# ----- 3. Preprocess Text Data -----
# Build the corpus from the 'content' column
corpus <- VCorpus(VectorSource(combined_df$content))

# Define a custom stopword list (standard stopwords + additional trivial words)
custom_stopwords <- c(stopwords("english"), "like", "just", "dont", "can", "will", "good", "really", "get", "way", "use","also","much")
#custom_stopwords <- c(stopwords("english"), "like", "just", "one", "thing", "ive", "year", "still", "well", "lot", "you're", "youre", "ur", "look", "think", "dont", "can", "will", "good", "really", "get", "way", "use","also","much")

# Apply text transformations
corpus <- tm_map(corpus, content_transformer(tolower))      # Lowercase
corpus <- tm_map(corpus, removePunctuation)                 # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                     # Remove numbers
corpus <- tm_map(corpus, removeWords, custom_stopwords)     # Remove custom stopwords
corpus <- tm_map(corpus, stripWhitespace)                   # Remove extra whitespace


# ----- 4. Convert the Corpus to Transactions -----
# Each document becomes a basket of unique words.
doc_list <- lapply(corpus, function(doc) {
  words <- unlist(strsplit(as.character(doc), "\\s+"))
  words <- words[words != ""]         # Remove empty strings
  words <- words[nchar(words) > 2]      # Remove words with 2 or fewer characters
  unique(words)
})
transactions <- as(doc_list, "transactions")

# Inspect a few transactions. 
trans_df <- as(transactions, "data.frame")
head(trans_df, 5)


# ----- 5. Run the Apriori Algorithm -----
# Adjust thresholds to get a smaller, more meaningful set of rules.
# (Support: fraction of documents; Confidence: rule accuracy
#  minlen: minimum number of items in the rule; maxlen: maximum rule length)
rules <- apriori(transactions, 
                 parameter = list(support = 0.05, 
                                  confidence = 0.5, 
                                  minlen = 4,
                                  maxlen = 5))

# Summarize the rules found
summary(rules)

# ----- 6. Sort and Inspect Top Rules -----
# Sort by different measures
rules_support    <- sort(rules, by = "support",    decreasing = TRUE)
rules_confidence <- sort(rules, by = "confidence", decreasing = TRUE)
rules_lift       <- sort(rules, by = "lift",       decreasing = TRUE)

# Extract top 15 rules for each measure
top15_support    <- head(rules_support, 15)
top15_confidence <- head(rules_confidence, 15)
top15_lift       <- head(rules_lift, 15)

cat("\n--- Top 15 Rules by Support ---\n")
arules::inspect(top15_support)

cat("\n--- Top 15 Rules by Confidence ---\n")
arules::inspect(top15_confidence)

cat("\n--- Top 15 Rules by Lift ---\n")
arules::inspect(top15_lift)

# ----- 7. Visualizations -----
# 7a. Network Graph of Top 15 Lift Rules
#plot(top15_lift, method = "graph", engine = "htmlwidget", 
#     main = "Network Graph: Top 15 Rules by Lift")

# 7b. Scatter Plot of All Rules (Support vs. Confidence, shading by Lift)
#plot(rules, measure = c("support", "confidence"), shading = "lift",
#     main = "Scatter Plot of Rules (Support vs. Confidence)")

