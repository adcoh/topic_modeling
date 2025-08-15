import typing as t
from collections import defaultdict
from itertools import combinations

import networkx as nx
import nltk
import polars as pl
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.stats import hypergeom

from WCSVNtm.logging_config import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)


# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))


# -- Step 1: Preprocessing ---------------------------------------------------
def preprocess_documents(documents: list[str]) -> list[list[list[str]]]:
    """Preprocess documents by tokenizing and cleaning text."""
    logger.info(f"Preprocessing {len(documents)} documents...")

    tokenized_docs = []
    total_sentences = 0
    total_words = 0

    for doc_id, doc in enumerate(documents):
        sentences = sent_tokenize(doc)
        clean_sentences = []
        doc_words = 0

        for sent in sentences:
            words = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
            # NOTE: Question whether we should clean stopwords or not?
            words = [w for w in words if w not in STOPWORDS]
            clean_sentences.append(words)
            doc_words += len(words)

        tokenized_docs.append(clean_sentences)
        total_sentences += len(sentences)
        total_words += doc_words

        if (doc_id + 1) % 10 == 0 or doc_id == len(documents) - 1:
            logger.info(f"Processed {doc_id + 1}/{len(documents)} documents")

    logger.info(
        f"Preprocessing complete: {total_sentences} sentences, {total_words} words"
    )
    return tokenized_docs


class SentenceId(t.NamedTuple):
    doc_id: int
    sentence_id: int

    def __str__(self) -> str:
        return f"D{self.doc_id}_S{self.sentence_id}"


# -- Step 2: Sentence-Word Bipartite & Word SVN Projection -------------------
def build_sentence_word_bipartite(
    tokenized_docs: list[list[list[str]]],
) -> tuple[nx.Graph, dict[str, set[SentenceId]]]:
    """Build bipartite graph between sentences and words."""
    logger.info("Building sentence-word bipartite graph...")

    B = nx.Graph()
    sentence_id = 0
    word_occurrence: dict[str, set[SentenceId]] = defaultdict(set)
    total_edges = 0

    for doc_id, sentences in enumerate(tokenized_docs):
        for sent in sentences:
            sid = SentenceId(doc_id, sentence_id)
            B.add_node(sid, bipartite="sentence", doc=doc_id)

            # Get unique words in sentence
            unique_words = set(sent)
            for word in unique_words:
                B.add_node(word, bipartite="word")
                B.add_edge(sid, word)
                word_occurrence[word].add(sid)
                total_edges += 1

            sentence_id += 1

            if sentence_id % 1000 == 0:
                logger.info(
                    f"Processed {sentence_id} sentences, {len(word_occurrence)} unique words"
                )

    logger.info(
        f"Bipartite graph complete: {sentence_id} sentences, {len(word_occurrence)} words, {total_edges} edges"
    )
    return B, word_occurrence


def statistically_validate_word_pairs(
    word_occurrence: dict[str, set[SentenceId]],
    total_sentences: int,
    alpha: float = 0.01,
) -> list[tuple[str, str]]:
    """
    Statistically validate word pairs using vectorized computation with Polars.

    This function is much faster than the original pairwise approach for large vocabularies.
    Falls back to optimized Python implementation if Polars is not available.
    """
    logger.info(
        f"Validating word pairs with {len(word_occurrence)} words, {total_sentences} sentences, alpha={alpha}"
    )

    try:
        logger.info("Using Polars implementation for vectorized computation")
        return _statistically_validate_word_pairs_polars(
            word_occurrence, total_sentences, alpha
        )
    except ImportError:
        logger.warning("Polars not available, falling back to Python implementation")
        return _statistically_validate_word_pairs_python(
            word_occurrence, total_sentences, alpha
        )


def _statistically_validate_word_pairs_polars(
    word_occurrence: dict[str, set[SentenceId]],
    total_sentences: int,
    alpha: float = 0.01,
) -> list[tuple[str, str]]:
    """Vectorized implementation using Polars for maximum performance."""
    words = list(word_occurrence.keys())
    if len(words) < 2:
        return []

    # Note: T will be calculated after filtering for non-zero intersections
    # to ensure Bonferroni correction is based on actual tests performed

    # Convert sets to lists for Polars compatibility
    word_data = []
    for word in words:
        # Convert set to sorted list for consistent hashing
        occurrence_list = sorted(map(str, word_occurrence[word]))
        word_data.append(
            {
                "word": word,
                "occurrence_list": occurrence_list,
                "count": len(occurrence_list),
            }
        )

    # Build occurrence matrix
    df = pl.DataFrame(word_data)

    # Generate all word pairs efficiently using a cross join
    # This creates a cartesian product of all words with all words
    # we
    pairs_df = (
        df.join(df, how="cross", suffix="_2")
        .select(
            [
                pl.col("word").alias("word1"),
                pl.col("word_2").alias("word2"),
                pl.col("occurrence_list").alias("set1"),
                pl.col("occurrence_list_2").alias("set2"),
            ]
        )
        .filter(pl.col("word1") != pl.col("word2"))
    )

    logger.info(f"Created DataFrame with {len(pairs_df)} pairs")

    # Join with occurrence data
    logger.info("Computing intersections and statistics...")
    interm_result = pairs_df.with_columns(
        [
            pl.col("set1")
            .list.set_intersection(pl.col("set2"))
            .list.len()
            .alias("Nij"),
            pl.col("set1").list.len().alias("Ni"),
            pl.col("set2").list.len().alias("Nj"),
        ]
    ).filter(pl.col("Nij") > 0)

    logger.info(f"Found {len(interm_result)} pairs with non-zero intersections")

    # Calculate T based on actual tests performed (pairs with non-zero intersections)
    T = len(interm_result)
    alpha_bonf = alpha / T
    logger.info(f"Bonferroni correction: T={T} tests, alpha_bonf={alpha_bonf:.6f}")

    # Compute p-values using numpy for speed
    logger.info("Computing p-values...")
    pval = interm_result.select(["Nij", "Ni", "Nj"]).to_numpy()
    pval = hypergeom.sf(pval[:, 0] - 1, total_sentences, pval[:, 1], pval[:, 2])

    interm_result = interm_result.with_columns([pl.Series("pval", pval)])
    result = interm_result.filter(pl.col("pval") < alpha_bonf)

    validated_pairs = len(result)
    logger.info(
        f"Statistical validation complete: {validated_pairs} pairs validated (p < {alpha_bonf:.6f})"
    )

    result = result.select(["word1", "word2"])
    return list(result.iter_rows())


def _statistically_validate_word_pairs_python(
    word_occurrence: dict[str, set[SentenceId]],
    total_sentences: int,
    alpha: float = 0.01,
) -> list[tuple[str, str]]:
    """Optimized Python implementation as fallback."""
    validated: list[tuple[str, str]] = []
    words = list(word_occurrence.keys())

    # First pass: count pairs with non-zero intersections to calculate T
    test_count = 0
    for wi, wj in combinations(words, 2):
        si, sj = word_occurrence[wi], word_occurrence[wj]
        Nij = len(si & sj)
        if Nij > 0:
            test_count += 1

    T = test_count
    alpha_bonf = alpha / T
    logger.info(f"Bonferroni correction: T={T} tests, alpha_bonf={alpha_bonf:.6f}")

    # Second pass: perform statistical tests
    for wi, wj in combinations(words, 2):
        si, sj = word_occurrence[wi], word_occurrence[wj]
        Nij = len(si & sj)
        if Nij <= 0:
            continue
        Ni, Nj = len(si), len(sj)
        pval = hypergeom.sf(Nij - 1, total_sentences, Ni, Nj)
        if pval < alpha_bonf:
            validated.append((wi, wj))

    return validated


def build_validated_word_network(edges: list[tuple[str, str]]) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


# -- Step 3: Document-Document SVN Projection -------------------------------
def build_doc_wordpair_bipartite(
    tokenized_docs: list[list[list[str]]], validated_word_pairs: list[tuple[str, str]]
) -> nx.Graph:
    B = nx.Graph()
    for doc_id, sentences in enumerate(tokenized_docs):
        B.add_node(f"doc_{doc_id}", bipartite="doc", doc=doc_id)
        pairs_in_doc = set()
        for sent in sentences:
            sent_set = set(sent)
            for wp in validated_word_pairs:
                if wp[0] in sent_set and wp[1] in sent_set:
                    pairs_in_doc.add(wp)
        for wp in pairs_in_doc:
            B.add_node(wp, bipartite="pair")
            B.add_edge(f"doc_{doc_id}", wp)
    return B


def statistically_validate_documents(
    doc_bipartite: nx.Graph, total_pairs: int, alpha: float = 0.01
) -> list[tuple[str, str, int]]:
    """
    Statistically validate document pairs using vectorized computation with Polars.

    This function is much faster than the original pairwise approach for large document sets.
    Falls back to optimized Python implementation if Polars is not available.
    """
    try:
        return _statistically_validate_documents_polars(
            doc_bipartite, total_pairs, alpha
        )
    except ImportError:
        return _statistically_validate_documents_python(
            doc_bipartite, total_pairs, alpha
        )


def _statistically_validate_documents_polars(
    doc_bipartite: nx.Graph, total_pairs: int, alpha: float = 0.01
) -> list[tuple[str, str, int]]:
    """Vectorized implementation using Polars for maximum performance."""
    docs = [n for n, d in doc_bipartite.nodes(data=True) if d["bipartite"] == "doc"]
    if len(docs) < 2:
        return []

    doc_neighbors: dict[str, set] = {d: set(doc_bipartite.neighbors(d)) for d in docs}
    # Note: T will be calculated after filtering for non-zero intersections
    # to ensure Bonferroni correction is based on actual tests performed

    # Build document neighbor matrix
    doc_matrix = pl.DataFrame(
        {"doc": docs, "neighbor_sets": [doc_neighbors[d] for d in docs]}
    ).lazy()

    # Generate all document pairs efficiently
    pairs_df = pl.DataFrame(
        {
            "doc1": [
                docs[i] for i in range(len(docs)) for j in range(i + 1, len(docs))
            ],
            "doc2": [
                docs[j] for i in range(len(docs)) for j in range(i + 1, len(docs))
            ],
        }
    ).lazy()

    # Join with neighbor data and compute intersections
    interm_result = (
        pairs_df.join(
            doc_matrix.select(["doc", "neighbor_sets"]).with_columns(
                pl.col("neighbor_sets").alias("set1")
            ),
            left_on="doc1",
            right_on="doc",
        )
        .join(
            doc_matrix.select(["doc", "neighbor_sets"]).with_columns(
                pl.col("neighbor_sets").alias("set2")
            ),
            left_on="doc2",
            right_on="doc",
        )
        .with_columns(
            [
                pl.col("set1").list.len().alias("Ki"),
                pl.col("set2").list.len().alias("Kj"),
                pl.col("set1")
                .list.set_intersection(pl.col("set2"))
                .list.len()
                .alias("Vij"),
            ]
        )
        .filter(pl.col("Vij") > 0)
    ).collect()

    # Calculate T based on actual tests performed (pairs with non-zero intersections)
    T = len(interm_result)
    alpha_bonf = alpha / T
    logger.info(
        f"Document validation Bonferroni correction: T={T} tests, alpha_bonf={alpha_bonf:.6f}"
    )

    # swapping to numpy for faster computation with scipy
    pval = interm_result.select(["Vij", "Ki", "Kj"]).to_numpy()
    pval = hypergeom.sf(pval[:, 0] - 1, total_pairs, pval[:, 1], pval[:, 2])
    result = interm_result.with_columns([pl.Series("pval", pval)])
    result = result.filter(pl.col("pval") < alpha_bonf)
    result = result.select(["doc1", "doc2", "Vij"])
    return list(result.iter_rows())


def _statistically_validate_documents_python(
    doc_bipartite: nx.Graph, total_pairs: int, alpha: float = 0.01
) -> list[tuple[str, str, int]]:
    """Optimized Python implementation as fallback."""
    docs = [n for n, d in doc_bipartite.nodes(data=True) if d["bipartite"] == "doc"]
    doc_neighbors: dict[str, set] = {d: set(doc_bipartite.neighbors(d)) for d in docs}
    validated: list[tuple[str, str, int]] = []

    # First pass: count pairs with non-zero intersections to calculate T
    test_count = 0
    for d1, d2 in combinations(docs, 2):
        s1, s2 = doc_neighbors[d1], doc_neighbors[d2]
        Vij = len(s1 & s2)
        if Vij > 0:
            test_count += 1

    T = test_count
    alpha_bonf = alpha / T
    logger.info(
        f"Document validation Bonferroni correction: T={T} tests, alpha_bonf={alpha_bonf:.6f}"
    )

    # Second pass: perform statistical tests
    for d1, d2 in combinations(docs, 2):
        s1, s2 = doc_neighbors[d1], doc_neighbors[d2]
        Vij = len(s1 & s2)
        if Vij <= 0:
            continue
        Ki, Kj = len(s1), len(s2)
        pval = hypergeom.sf(Vij - 1, total_pairs, Ki, Kj)
        if pval < alpha_bonf:
            validated.append((d1, d2, Vij))
    return validated


def build_validated_doc_network(
    validated_doc_edges: list[tuple[str, str, int]],
) -> nx.Graph:
    G = nx.Graph()
    for u, v, w in validated_doc_edges:
        G.add_edge(u, v, weight=w)
    return G


# -- Step 4: Community Detection with Leiden fallback to Louvain -----------
def community_detection(
    G: nx.Graph,
    use_leiden: bool = True,
    weight: str = "weight",
    resolution: int = 1,
    seed: int | None = None,
) -> list[set]:
    try:
        return list(
            nx.algorithms.community.leiden_communities(  # type: ignore
                G, backend="parallel", weight=weight, resolution=resolution, seed=seed
            )
        )
    except (ImportError, nx.NetworkXNotImplemented, NotImplementedError):
        return t.cast(
            list[set[str]],
            nx.algorithms.community.louvain_communities(
                G, weight=weight, resolution=resolution, seed=seed
            ),
        )


# -- Step 5: Topic-Document Association via Fisher's Exact & FDR -------------
def associate_topics_documents(
    G_word: nx.Graph, doc_clusters: dict[str, list], fdr_alpha: float = 0.05
) -> dict[int, list[str]]:
    universe = set(G_word.nodes())
    associations: dict[int, list[str]] = defaultdict(list)
    tests: list[tuple[int, str, float]] = []
    # TODO: vectorize this computation in polars
    for topic_id, topic_words in enumerate(doc_clusters["topics"]):
        for doc in doc_clusters["docs"]:
            doc_words = doc_clusters["doc_to_words"][doc]
            overlap = len(topic_words & doc_words)
            K = len(topic_words)
            N = len(universe)
            n = len(doc_words)
            pval = hypergeom.sf(overlap - 1, N, K, n)
            tests.append((topic_id, doc, pval))  # type: ignore
    tests_sorted = sorted(tests, key=lambda x: x[2])
    m = len(tests)
    for rank, (topic_id, doc, pval) in enumerate(tests_sorted, start=1):
        if pval <= (rank / m) * fdr_alpha:
            associations[topic_id].append(doc)
    return associations


# -- Step 6: Topic Importance via Modularity Contribution ------------------
def compute_topic_importance(
    G_word: nx.Graph, topics: list[set[str]]
) -> dict[int, dict[str, float]]:
    """
    Compute topic importance using modularity contribution as described in the WCSVNtm paper.

    For each word i in a community c, the modularity contribution is:
    M_i = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] for all j in community c, j ≠ i

    Word importance is then normalized: I_i = M_i / Σ(M_k) for all k in community c

    Args:
        G_word: The word co-occurrence network
        topics: List of word communities (topics) as sets of word strings

    Returns:
        Dictionary mapping topic index to word importance scores within that topic
    """
    topic_importance: dict[int, dict[str, float]] = {}

    # Total number of edges in the network (m in the paper)
    total_edges: int = G_word.number_of_edges()
    if total_edges == 0:
        logger.warning("Graph has no edges, returning empty importance scores")
        return {i: {} for i in range(len(topics))}

    # Precompute node degrees for efficiency
    node_degrees: dict[str, int] = dict(G_word.degree())  # type: ignore

    # Convert graph to adjacency dictionary for fast edge lookups
    adjacency_dict: dict[str, dict[str, dict]] = nx.to_dict_of_dicts(G_word)  # type: ignore

    for topic_idx, community in enumerate(topics):
        # Skip empty communities
        if not community:
            topic_importance[topic_idx] = {}
            continue

        # Dictionary to store modularity contributions for each word in this topic
        modularity_contributions: dict[str, float] = {}

        # Compute modularity contribution for each word in the community
        for word_i in community:
            # Initialize modularity contribution for word i
            modularity_contrib_i: float = 0.0

            # Sum over all other words j in the same community
            for word_j in community:
                if word_i == word_j:
                    continue

                # Check if edge exists between word_i and word_j
                edge_exists: int = 1 if word_j in adjacency_dict.get(word_i, {}) else 0

                # Expected number of edges under null model
                expected_edges: float = (
                    node_degrees[word_i] * node_degrees[word_j]
                ) / (2 * total_edges)

                # Add contribution to modularity
                modularity_contrib_i += edge_exists - expected_edges

            # Apply the (1/2m) normalization factor from the paper
            modularity_contrib_i /= 2 * total_edges
            modularity_contributions[word_i] = modularity_contrib_i

        # Normalize contributions to get importance scores
        total_modularity: float = sum(modularity_contributions.values())

        if total_modularity <= 0:
            # If total modularity is zero or negative, assign equal importance
            logger.warning(
                f"Topic {topic_idx} has zero or negative total modularity, assigning equal weights"
            )
            equal_weight: float = 1.0 / len(community) if community else 0.0
            topic_importance[topic_idx] = {word: equal_weight for word in community}
        else:
            # Normalize by total modularity to get relative importance
            topic_importance[topic_idx] = {
                word: contrib / total_modularity
                for word, contrib in modularity_contributions.items()
            }

    return topic_importance


# -- Full Pipeline -----------------------------------------------------------
def run_wcsvntm(
    documents: list[str], alpha_word: float = 0.01, alpha_doc: float = 0.01
) -> dict[str, t.Any]:
    """Run the complete WCSVNtm pipeline."""
    logger.info("=" * 60)
    logger.info("Starting WCSVNtm Pipeline")
    logger.info("=" * 60)
    logger.info(f"Parameters: alpha_word={alpha_word}, alpha_doc={alpha_doc}")

    # Step 1: Preprocessing
    logger.info("\n--- Step 1: Document Preprocessing ---")
    tokenized = preprocess_documents(documents)

    # Step 2: Build bipartite graph
    logger.info("\n--- Step 2: Building Bipartite Graph ---")
    B_sw, word_occ = build_sentence_word_bipartite(tokenized)
    total_sent = len(
        [n for n, d in B_sw.nodes(data=True) if d["bipartite"] == "sentence"]
    )
    logger.info(f"Total sentences in graph: {total_sent}")

    # Step 3: Word pair validation
    logger.info("\n--- Step 3: Word Pair Validation ---")
    wpairs = statistically_validate_word_pairs(word_occ, total_sent, alpha_word)
    logger.info(f"Validated word pairs: {len(wpairs)}")

    # Step 4: Build word network
    logger.info("\n--- Step 4: Building Word Network ---")
    G_word = build_validated_word_network(wpairs)
    logger.info(
        f"Word network: {G_word.number_of_nodes()} nodes, {G_word.number_of_edges()} edges"
    )

    # Step 5: Document-word pair bipartite
    logger.info("\n--- Step 5: Building Document-Word Pair Bipartite ---")
    B_dp = build_doc_wordpair_bipartite(tokenized, wpairs)
    total_pairs = len(list(B_dp.nodes())) - len(documents)
    logger.info(
        f"Document-word pair bipartite: {len(documents)} docs, {total_pairs} word pairs"
    )

    # Step 6: Document validation
    logger.info("\n--- Step 6: Document Validation ---")
    doc_edges = statistically_validate_documents(B_dp, total_pairs, alpha_doc)
    logger.info(f"Validated document pairs: {len(doc_edges)}")

    # Step 7: Build document network
    logger.info("\n--- Step 7: Building Document Network ---")
    G_doc = build_validated_doc_network(doc_edges)
    logger.info(
        f"Document network: {G_doc.number_of_nodes()} nodes, {G_doc.number_of_edges()} edges"
    )

    # Step 8: Community detection
    logger.info("\n--- Step 8: Community Detection ---")
    topics = community_detection(G_word)
    doc_clusters = community_detection(G_doc)
    logger.info(f"Word communities: {len(topics)}")
    logger.info(f"Document communities: {len(doc_clusters)}")

    # Step 9: Topic-document association
    logger.info("\n--- Step 9: Topic-Document Association ---")
    doc_to_words = {
        f"doc_{i}": {w for sent in tokenized[i] for w in sent if w in G_word}
        for i in range(len(documents))
    }
    association = associate_topics_documents(
        G_word,
        {
            "topics": topics,
            "docs": list(doc_to_words.keys()),
            "doc_to_words": doc_to_words,  # type: ignore
        },
    )
    logger.info(f"Topic-document associations computed for {len(association)} topics")

    # Step 10: Topic importance
    logger.info("\n--- Step 10: Computing Topic Importance ---")
    importance = compute_topic_importance(G_word, topics)
    logger.info("Topic importance computation complete")

    logger.info("\n" + "=" * 60)
    logger.info("WCSVNtm Pipeline Complete!")
    logger.info("=" * 60)

    return {
        "word_graph": G_word,
        "doc_graph": G_doc,
        "topics": topics,
        "doc_clusters": doc_clusters,
        "associations": association,
        "importance": importance,
    }


if __name__ == "__main__":
    # Configure logging for demo
    from WCSVNtm.data_io import DataReader

    setup_logging(level="INFO")

    logger.info("Running WCSVNtm demo with sample documents...")

    docs = DataReader().read_all_txt_files()

    try:
        results = run_wcsvntm(docs)

        logger.info("\n" + "=" * 40)
        logger.info("DEMO RESULTS")
        logger.info("=" * 40)
        logger.info(f"Topics found: {len(results['topics'])}")
        for i, topic in enumerate(results["topics"]):
            logger.info(f"Topic {i}: {list(topic)[:5]}...")  # Show first 5 words

        logger.info(f"Document clusters: {len(results['doc_clusters'])}")
        logger.info(f"Topic-document associations: {len(results['associations'])}")

        if results["importance"]:
            logger.info(
                f"Topic importance computed for {len(results['importance'])} topics"
            )

        logger.info("Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
