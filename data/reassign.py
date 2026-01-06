import numpy as np
import pandas as pd

def reassign_patient_ids_with_repeats(
    encounters_csv="encounters_synth.csv",
    out_csv="encounters_synth_with_repeats.csv",
    seed=42,
    # target average encounters per patient ~ 1.25–1.35 depending on weights
    enc_per_patient_choices=(1, 2, 3, 4, 5),
    enc_per_patient_weights=(0.0625, 0.25, 0.375, 0.25, 0.0625),
):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(encounters_csv, dtype=str)

    n_enc = len(df)
    # how many encounters each synthetic patient will have
    counts = []
    while sum(counts) < n_enc:
        c = rng.choice(enc_per_patient_choices, p=np.array(enc_per_patient_weights)/np.sum(enc_per_patient_weights))
        counts.append(int(c))
    # trim to exactly n_enc
    total = sum(counts)
    if total > n_enc:
        over = total - n_enc
        # reduce from the end by 'over' (guaranteed not to go below 1 if we do carefully)
        i = len(counts) - 1
        while over > 0 and i >= 0:
            dec = min(over, counts[i] - 1)
            counts[i] -= dec
            over -= dec
            i -= 1

    num_patients = len(counts)

    # generate patient ids (obfuscated string, sometimes decimals)
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    def make_pid():
        base = "".join(rng.choice(alphabet, size=rng.integers(8, 13)))
        if rng.random() < 0.35:
            return f"{base}.{rng.integers(0, 100)}"
        return base

    patient_ids = [make_pid() for _ in range(num_patients)]

    # assign encounters to patients
    idx = np.arange(n_enc)
    rng.shuffle(idx)

    new_pid = np.empty(n_enc, dtype=object)
    cursor = 0
    for pid, c in zip(patient_ids, counts):
        chunk = idx[cursor:cursor+c]
        new_pid[chunk] = pid
        cursor += c

    df["PAITENT_ID"] = new_pid
    df.to_csv(out_csv, index=False)

    # quick sanity check printout
    grp = df.groupby("PAITENT_ID").size()
    print("encounters:", n_enc)
    print("patients:", len(grp))
    print("mean encounters/patient:", grp.mean())
    print("distribution:\n", grp.value_counts().sort_index())

    return df

df2 = reassign_patient_ids_with_repeats()
