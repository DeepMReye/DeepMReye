import os
import json
import numpy as np
import pandas as pd
from deepmreye.evaluate.eval_probe import (
    load_all_dataset_meta,
    load_preprocessed_fold_data,
    fit_ridge_and_eval
)
from deepmreye.evaluate.features import OrbitExtractor

def main():
    print("=" * 80)
    print("[*] LOW-SAMPLE DATA EFFICIENCY BENCHMARK: CORPUS-PCA:64 VS FOLD-PCA:64")
    print("=" * 80)
    
    meta = load_all_dataset_meta()
    test_folds = meta['test_folds']
    
    sample_budgets = [100, 250, 500, 1000, 2500, 5000, "All"]
    
    results = {}
    
    for fold_idx, fold_name in enumerate(test_folds):
        print(f"\n[{fold_idx+1}/{len(test_folds)}] --- Evaluating Fold: {fold_name} ---")
        
        train_data, test_data = load_preprocessed_fold_data(fold_name, meta)
        
        # Extract 64D features for corpus-pca and fold-pca
        corpus_ext = OrbitExtractor(basis_name="corpus-pca", n_components=64)
        fold_ext = OrbitExtractor(basis_name="fold-pca", n_components=64)
        
        X_train_corpus, y_train, train_sub_ids = corpus_ext(train_data, is_train=True)
        X_test_corpus, y_test, test_sub_ids = corpus_ext(test_data, is_train=False)
        
        X_train_fold, _, _ = fold_ext(train_data, is_train=True)
        X_test_fold, _, _ = fold_ext(test_data, is_train=False)
        
        fold_results = {}
        
        total_train_samples = len(y_train)
        print(f"    Total available training frames: {total_train_samples}")
        print(f"    {'Budget (N)':<12} | {'Corpus-PCA (r)':<15} | {'Fold-PCA (r)':<15} | {'Diff (Δr)':<12} | {'Winner':<12}")
        print("    " + "-" * 72)
        
        for budget in sample_budgets:
            if budget == "All" or budget >= total_train_samples:
                n_samples = total_train_samples
                budget_label = f"All ({n_samples})"
            else:
                n_samples = budget
                budget_label = str(n_samples)
                
            # Deterministic subset of first n_samples
            X_tr_c_sub = X_train_corpus[:n_samples]
            X_tr_f_sub = X_train_fold[:n_samples]
            y_tr_sub = y_train[:n_samples]
            
            # Fit and evaluate RidgeCV
            res_corpus = fit_ridge_and_eval(X_tr_c_sub, y_tr_sub, X_test_corpus, y_test, test_sub_ids)
            res_fold = fit_ridge_and_eval(X_tr_f_sub, y_tr_sub, X_test_fold, y_test, test_sub_ids)
            
            r_c = res_corpus['r_mean']
            r_f = res_fold['r_mean']
            diff = r_c - r_f
            
            winner = "CORPUS-PCA 🏆" if diff > 0.001 else ("FOLD-PCA" if diff < -0.001 else "TIE")
            
            print(f"    {budget_label:<12} | {r_c:<15.4f} | {r_f:<15.4f} | {diff:<+12.4f} | {winner:<12}")
            
            fold_results[str(budget)] = {
                "n_samples": n_samples,
                "corpus_pca_r_mean": float(r_c),
                "corpus_pca_r_x": float(res_corpus['r_x']),
                "corpus_pca_r_y": float(res_corpus['r_y']),
                "corpus_pca_median_r": float(res_corpus['median_r']),
                "fold_pca_r_mean": float(r_f),
                "fold_pca_r_x": float(res_fold['r_x']),
                "fold_pca_r_y": float(res_fold['r_y']),
                "fold_pca_median_r": float(res_fold['median_r']),
                "diff_r_mean": float(diff),
                "winner": winner
            }
            
            if budget != "All" and n_samples == total_train_samples:
                # Stop redundant sweeps if budget exceeds dataset size
                break
                
        results[fold_name] = fold_results
        
    # Aggregate results across datasets
    os.makedirs("results/low_sample_benchmark", exist_ok=True)
    out_file = "results/low_sample_benchmark/corpus_vs_fold_pca_low_sample.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"[*] LOW-SAMPLE BENCHMARK COMPLETE! Results saved to {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
