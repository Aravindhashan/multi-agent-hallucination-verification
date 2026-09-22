"""Submodule 3.7 Stress Test: Domain-Diverse Decomposition Quality Benchmark.

Evaluates NLPClaimExtractor and DecompositionQualityEvaluator across 60 domain-diverse sentences
spanning 5 non-aerospace domains:
1. Medical & Clinical
2. Business & Finance
3. Sports & Entertainment
4. Consumer Products & Tech Hardware
5. Arts, History & Law

Asserts atomicity, coverage, and entailment metrics across all domains.
"""

import pytest
from app.services.nlp_claim_extractor import NLPClaimExtractor
from app.services.decomposition_evaluator import DecompositionQualityEvaluator


DOMAINS_DATA: dict[str, list[str]] = {
    "Medical & Clinical": [
        "Pfizer developed the Comirnaty vaccine, approved by the FDA in August 2021, to prevent COVID-19.",
        "Penicillin was discovered in 1928 by Alexander Fleming at St. Mary's Hospital in London.",
        "Aspirin, synthesized in 1897 by Felix Hoffmann at Bayer, relieves mild pain and reduces fever.",
        "Insulin was first administered to humans in January 1922 by Frederick Banting and Charles Best.",
        "The clinical trial enrolled 450 patients across 12 medical centers in Germany.",
        "CRISPR-Cas9 gene editing, pioneered in 2012 by Jennifer Doudna and Emmanuelle Charpentier, treats sickle cell anemia.",
        "Metformin lowers blood glucose levels in patients with type 2 diabetes.",
        "The polio vaccine, developed in 1953 by Jonas Salk at the University of Pittsburgh, was declared safe in 1955.",
        "Remdesivir received emergency use authorization in May 2020.",
        "Statins reduce low-density lipoprotein cholesterol by approximately 30 percent.",
        "The human genome contains approximately 3 billion base pairs across 23 chromosome pairs.",
        "Dr. James Allison won the Nobel Prize in 2018 for cancer immunotherapy discoveries.",
    ],
    "Business & Finance": [
        "Apple Inc. reported quarterly revenue of 90 billion dollars in July 2023.",
        "Microsoft acquired LinkedIn in December 2016 for 26 billion dollars.",
        "Tesla opened its Gigafactory near Berlin in March 2022.",
        "The Federal Reserve raised the benchmark interest rate to 5.25 percent in July 2023.",
        "Amazon was founded in July 1994 by Jeff Bezos in Washington.",
        "Goldman Sachs completed the merger of two European energy firms in 2021.",
        "The company generated 4.5 billion dollars in free cash flow during fiscal year 2022.",
        "Warren Buffett purchased 400 million shares of Coca-Cola through Berkshire Hathaway.",
        "Alphabet announced a 70 billion dollar share repurchase program in April 2024.",
        "JPMorgan Chase operates in over 100 countries worldwide.",
        "The consumer price index rose 3.2 percent over the twelve months ending in July 2023.",
        "Nvidia's market capitalization surpassed 2 trillion dollars in February 2024.",
    ],
    "Sports & Entertainment": [
        "Usain Bolt set the world record of 9.58 seconds in Berlin in August 2009.",
        "Lionel Messi won his eighth Ballon d'Or in Paris in October 2023.",
        "The film Titanic, directed in 1997 by James Cameron, won 11 Academy Awards.",
        "Michael Phelps won 8 gold medals at the 2008 Beijing Olympics.",
        "The Beatles released Abbey Road in September 1969 through Apple Records.",
        "Serena Williams won 23 Grand Slam singles titles during her career.",
        "The FIFA World Cup was held in Qatar in November 2022.",
        "Christopher Nolan directed Oppenheimer, released in July 2023, grossing 950 million dollars.",
        "Roger Federer won 8 Wimbledon championships between 2003 and 2017.",
        "The Louvre Museum in Paris welcomed 8.9 million visitors in 2023.",
        "Simone Biles won 4 gold medals at the 2016 Rio Olympics.",
        "Steven Spielberg directed Jurassic Park, released in June 1993 by Universal Pictures.",
    ],
    "Consumer Products & Tech Hardware": [
        "The iPhone 15 Pro features an A17 Pro chip and weighs 187 grams.",
        "The Sony PlayStation 5 was released in November 2020 with 825 gigabytes of storage.",
        "The MacBook Air contains an M3 processor and provides up to 18 hours of battery life.",
        "Nintendo released the Switch console in March 2017, selling over 130 million units.",
        "The electric vehicle travels 300 miles on a single charge of its 75 kilowatt-hour battery.",
        "Dyson launched its Supersonic hair dryer in 2016 with a 1600-watt digital motor.",
        "The Samsung Galaxy S24 Ultra includes a 200-megapixel primary camera sensor.",
        "The Canon EOS R5 camera captures 8K video at 30 frames per second.",
        "The smart thermostat reduces heating costs by approximately 15 percent.",
        "The drone weighs 249 grams and features a 4K camera with 3-axis stabilization.",
        "The mechanical keyboard uses Cherry MX switches rated for 50 million keystrokes.",
        "Intel released its Core i9 processor in October 2023 with 24 computing cores.",
    ],
    "Arts, History & Law": [
        "The Treaty of Paris, signed in September 1783 by Benjamin Franklin, ended the American Revolutionary War.",
        "Leonardo da Vinci painted the Mona Lisa in Florence around 1503.",
        "The United States Constitution was signed in Philadelphia on September 17, 1787.",
        "The Supreme Court ruled in Brown v. Board of Education in May 1954.",
        "The Magna Carta was signed in June 1215 by King John at Runnymede.",
        "Michelangelo completed the Sistine Chapel ceiling in Rome in October 1512.",
        "The Emancipation Proclamation was issued on January 1, 1863 by Abraham Lincoln.",
        "The Parthenon was built in Athens between 447 BC and 432 BC.",
        "The United Nations was established in San Francisco on October 24, 1945.",
        "William Shakespeare wrote Hamlet around 1600 in London.",
        "The 19th Amendment granted women the right to vote in August 1920.",
        "The Sydney Opera House, designed by Jorn Utzon, opened in October 1973 in Sydney.",
    ],
}


def test_domain_diverse_decomposition_benchmark():
    """Run full 60-sentence evaluation and report metrics across all 5 domains."""
    extractor = NLPClaimExtractor()
    evaluator = DecompositionQualityEvaluator()

    report = evaluator.evaluate_corpus(DOMAINS_DATA, extractor)
    domain_scores = report["domains"]
    aggregate = report["aggregate"]

    for domain_name, scores in domain_scores.items():
        # Assert per-domain minimum benchmarks (Rank 4 & 5 criteria)
        assert scores["mean_coverage"] >= 0.80, f"{domain_name} mean coverage {scores['mean_coverage']} < 0.80"
        assert scores["mean_atomicity"] >= 0.85, f"{domain_name} mean atomicity {scores['mean_atomicity']} < 0.85"
        assert scores["mean_entailment"] >= 0.75, f"{domain_name} mean entailment {scores['mean_entailment']} < 0.75"
        assert scores["high_quality_rate"] >= 0.85, f"{domain_name} HQ rate {scores['high_quality_rate']} < 0.85"

    assert aggregate["total_sentences"] == 60
    assert aggregate["total_claims"] >= 60

    # Strict aggregate benchmarks
    assert aggregate["mean_coverage"] >= 0.95, f"Aggregate coverage {aggregate['mean_coverage']} < 0.95"
    assert aggregate["mean_atomicity"] >= 0.95, f"Aggregate atomicity {aggregate['mean_atomicity']} < 0.95"
    assert aggregate["mean_entailment"] >= 0.95, f"Aggregate entailment {aggregate['mean_entailment']} < 0.95"
    assert aggregate["high_quality_rate"] >= 0.95, f"Aggregate HQ rate {aggregate['high_quality_rate']} < 0.95"

    # Print markdown table for audit logging
    print("\n" + "=" * 80)
    print("DOMAIN-DIVERSE DECOMPOSITION BENCHMARK (60 SENTENCES / 5 DOMAINS)")
    print("=" * 80)
    print(f"{'Domain':<35} | {'Coverage':<10} | {'Atomicity':<10} | {'Entailment':<10} | {'HQ Rate':<10}")
    print("-" * 85)
    for d, s in domain_scores.items():
        print(f"{d:<35} | {s['mean_coverage']:<10} | {s['mean_atomicity']:<10} | {s['mean_entailment']:<10} | {s['high_quality_rate']:<10}")
    print("-" * 85)
    print(f"{'AGGREGATE BENCHMARK':<35} | {aggregate['mean_coverage']:<10} | {aggregate['mean_atomicity']:<10} | {aggregate['mean_entailment']:<10} | {aggregate['high_quality_rate']:<10}")
    print("=" * 80)


def test_evaluator_evaluate_batch_empty_and_valid():
    """Verify evaluate_batch on empty and known batch inputs."""
    evaluator = DecompositionQualityEvaluator()

    # Empty batch
    empty_res = evaluator.evaluate_batch([])
    assert empty_res["total_items"] == 0
    assert empty_res["mean_coverage"] == 0.0

    # Valid batch
    from app.services.claim_models import AtomicClaim
    claims = [
        AtomicClaim(claim_id="c1", claim_text="Paris is the capital of France."),
    ]
    batch_res = evaluator.evaluate_batch([("Paris is the capital of France.", claims)])
    assert batch_res["total_items"] == 1
    assert batch_res["mean_coverage"] > 0.80
    assert batch_res["mean_atomicity"] == 1.0
    assert batch_res["high_quality_rate"] == 1.0
