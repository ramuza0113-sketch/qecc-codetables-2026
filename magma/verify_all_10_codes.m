// ==========================================================================
// Magma Verification Script for 10 Explicit Quantum Error-Correcting Codes
// Improving 16 Entries in CodeTables.de (Katsutoshi Hashimoto, Nov 2026)
// Run with: magma verify_all_10_codes.m
// ==========================================================================

P2<x> := PolynomialRing(GF(2)); F4<w> := ext<GF(2) | x^2 + x + 1>;
P3<x> := PolynomialRing(GF(3)); F9<t> := ext<GF(3) | x^2 + 1>;
P2<x> := PolynomialRing(GF(2)); F16<z> := ext<GF(2) | x^4 + x + 1>;
P5<x> := PolynomialRing(GF(5)); F25<t> := ext<GF(5) | x^2 + 3>;
P7<x> := PolynomialRing(GF(7)); F49<t> := ext<GF(7) | x^2 + 4>;

// --- [[36, 26, 4]]_2 (CodeTables.de was 3..4) ---
G_2_36_26 := Matrix(F4, 5, 36, [1, 0, 0, 0, 0, w, w, 1, 0, 0, w, 1+w, 1+w, 1, 0, 1+w, 1+w, 1, 1+w, 0, 0, 1, 0, 1, w, w, 1, 1+w, 1, w, 1, 1, 0, w, 0, 1, 0, 1, 0, 0, 0, 0, w, w, 1, 0, 0, w, 1+w, 1+w, 1, 0, 1+w, 1+w, 1, 1+w, w, 0, 1, 0, 1, w, w, 1, 1+w, 1, 0, 1, 1, 0, w, 1, 0, 0, 1, 0, 0, 0, 0, w, w, 1, 1, 0, w, 1+w, 1+w, 1+w, 0, 1+w, 1+w, 1, 1, w, 0, 1, 0, 1, w, w, 1, 1+w, w, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, w, w, 1+w, 1, 0, w, 1+w, 1, 1+w, 0, 1+w, 1+w, 0, 1, w, 0, 1, 1+w, 1, w, w, 1, 0, w, 0, 1, 1, 1, 0, 0, 0, 0, 1, w, 1, 0, 0, w, 1+w, 1+w, 1, 0, w, 1+w, 1, 1+w, 0, 1+w, 1, 0, 1, w, 0, 1, 1+w, 1, w, w, 1, 0, w, 0, 1, 1]);
C_2_36_26 := LinearCode(G_2_36_26);
assert Dimension(C_2_36_26) eq 5;
assert C_2_36_26 subset HermitianDual(C_2_36_26);
d_dual_2_36_26 := MinimumWeight(HermitianDual(C_2_36_26));
assert d_dual_2_36_26 ge 4;
Q_2_36_26 := QuantumCode(C_2_36_26);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_2_36_26), Dimension(Q_2_36_26), 4, 2, d_dual_2_36_26;

// --- [[49, 33, 5]]_2 (CodeTables.de was 4..6) ---
G_2_49_33 := Matrix(F4, 8, 49, [1, 0, 0, 0, 0, 0, 0, 0, 1+w, 1, 1+w, 1+w, 1+w, 1, 1, w, 0, 0, w, 0, 1, 1, 1+w, 1+w, 1, w, 1, 0, 1, 0, 0, w, w, w, 1, 0, 0, w, 1+w, w, w, 0, 0, w, w, w, w, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, w, 1+w, 1, 1+w, 1+w, 1+w, 1, 1, 1+w, 0, 0, w, 0, 1, 1, 1+w, w, 1, w, 1, 0, 1, 0, 0, w, w, w, 1, 0, 0, w, 1+w, 1, w, 0, 0, w, w, w, w, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, w, 1+w, 1, 1+w, 1+w, 1+w, 1, 1+w, 1+w, 0, 0, w, 0, 1, 1, 0, w, 1, w, 1, 0, 1, 0, 1+w, w, w, w, 1, 0, 0, w, w, 1, w, 0, 0, w, w, w, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, w, 1+w, 1, 1+w, 1+w, 1+w, 1, 1+w, 1+w, 0, 0, w, 0, 1, 0, 0, w, 1, w, 1, 0, 1, w, 1+w, w, w, w, 1, 0, 0, w, w, 1, w, 0, 0, w, w, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1+w, 1, 1, w, 1+w, 1, 1+w, 1+w, 1, 1, 1+w, 1+w, 0, 0, w, 0, 1, 0, 0, w, 1, w, 1, 0, 0, w, 1+w, w, w, w, 1, 0, w, w, w, 1, w, 0, 0, w, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1+w, 1+w, 1, 1, w, 1+w, 1, 1+w, 0, 1, 1, 1+w, 1+w, 0, 0, w, 0, 1, 0, 0, w, 1, w, 1, 0, 0, w, 1+w, w, w, w, 1, w, w, w, w, 1, w, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1+w, 1+w, 1+w, 1, 1, w, 1+w, 1, w, 0, 1, 1, 1+w, 1+w, 0, 0, 1, 0, 1, 0, 0, w, 1, w, 1, 0, 0, w, 1+w, w, w, w, 0, w, w, w, w, 1, w, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1+w, 1+w, 1+w, 1, 1, w, 1+w, 0, w, 0, 1, 1, 1+w, 1+w, 0, w, 1, 0, 1, 0, 0, w, 1, w, 1, 0, 0, w, 1+w, w, w, 0, 0, w, w, w, w, 1, w, 1]);
C_2_49_33 := LinearCode(G_2_49_33);
assert Dimension(C_2_49_33) eq 8;
assert C_2_49_33 subset HermitianDual(C_2_49_33);
d_dual_2_49_33 := MinimumWeight(HermitianDual(C_2_49_33));
assert d_dual_2_49_33 ge 5;
Q_2_49_33 := QuantumCode(C_2_49_33);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_2_49_33), Dimension(Q_2_49_33), 5, 2, d_dual_2_49_33;

// --- [[20, 10, 5]]_3 (CodeTables.de was 4..5) ---
G_3_20_10 := Matrix(F9, 5, 20, [1, 0, 0, 0, 0, 1+t, 1, 1+t, 1, 1+2*t, 1+t, 2, 2, 2+2*t, 1+t, 1+t, 0, 1+t, 2*t, 2+2*t, 0, 1, 0, 0, 0, 1+2*t, 1+t, 1, 1+t, 1, 1+t, 1+t, 2, 2, 2+2*t, 2+2*t, 1+t, 0, 1+t, 2*t, 0, 0, 1, 0, 0, 1, 1+2*t, 1+t, 1, 1+t, 2+2*t, 1+t, 1+t, 2, 2, 2*t, 2+2*t, 1+t, 0, 1+t, 0, 0, 0, 1, 0, 1+t, 1, 1+2*t, 1+t, 1, 2, 2+2*t, 1+t, 1+t, 2, 1+t, 2*t, 2+2*t, 1+t, 0, 0, 0, 0, 0, 1, 1, 1+t, 1, 1+2*t, 1+t, 2, 2, 2+2*t, 1+t, 1+t, 0, 1+t, 2*t, 2+2*t, 1+t]);
C_3_20_10 := LinearCode(G_3_20_10);
assert Dimension(C_3_20_10) eq 5;
assert C_3_20_10 subset HermitianDual(C_3_20_10);
d_dual_3_20_10 := MinimumWeight(HermitianDual(C_3_20_10));
assert d_dual_3_20_10 ge 5;
Q_3_20_10 := QuantumCode(C_3_20_10);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_3_20_10), Dimension(Q_3_20_10), 5, 3, d_dual_3_20_10;

// --- [[18, 6, 6]]_4 (CodeTables.de was 5..7) ---
G_4_18_6 := Matrix(F16, 6, 18, [1, 0, 0, 0, 0, 0, 1+z+z^3, z+z^2, z^2, 1+z^3, 1+z+z^3, 0, z+z^2, z^3, z+z^3, z+z^2, 1, z+z^2, 0, 1, 0, 0, 0, 0, 0, 1+z+z^3, z+z^2, z^2, 1+z^3, 1+z+z^3, z+z^2, z+z^2, z^3, z+z^3, z+z^2, 1, 0, 0, 1, 0, 0, 0, 1+z+z^3, 0, 1+z+z^3, z+z^2, z^2, 1+z^3, 1, z+z^2, z+z^2, z^3, z+z^3, z+z^2, 0, 0, 0, 1, 0, 0, 1+z^3, 1+z+z^3, 0, 1+z+z^3, z+z^2, z^2, z+z^2, 1, z+z^2, z+z^2, z^3, z+z^3, 0, 0, 0, 0, 1, 0, z^2, 1+z^3, 1+z+z^3, 0, 1+z+z^3, z+z^2, z+z^3, z+z^2, 1, z+z^2, z+z^2, z^3, 0, 0, 0, 0, 0, 1, z+z^2, z^2, 1+z^3, 1+z+z^3, 0, 1+z+z^3, z^3, z+z^3, z+z^2, 1, z+z^2, z+z^2]);
C_4_18_6 := LinearCode(G_4_18_6);
assert Dimension(C_4_18_6) eq 6;
assert C_4_18_6 subset HermitianDual(C_4_18_6);
d_dual_4_18_6 := MinimumWeight(HermitianDual(C_4_18_6));
assert d_dual_4_18_6 ge 6;
Q_4_18_6 := QuantumCode(C_4_18_6);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_4_18_6), Dimension(Q_4_18_6), 6, 4, d_dual_4_18_6;

// --- [[19, 7, 6]]_4 (CodeTables.de was 5..7) ---
G_4_19_7 := Matrix(F16, 6, 19, [1, 0, 0, 0, 0, 0, 1, 1+z^2, 1+z^2, 1+z, z^3, 1+z+z^2+z^3, 0, 1+z+z^3, 1+z+z^2, z, 1+z+z^2, z+z^2+z^3, 1, 0, 1, 0, 0, 0, 0, 1+z+z^2+z^3, 1, 1+z^2, 1+z^2, 1+z, z^3, z+z^2+z^3, 0, 1+z+z^3, 1+z+z^2, z, 1+z+z^2, 1, 0, 0, 1, 0, 0, 0, z^3, 1+z+z^2+z^3, 1, 1+z^2, 1+z^2, 1+z, 1+z+z^2, z+z^2+z^3, 0, 1+z+z^3, 1+z+z^2, z, 1, 0, 0, 0, 1, 0, 0, 1+z, z^3, 1+z+z^2+z^3, 1, 1+z^2, 1+z^2, z, 1+z+z^2, z+z^2+z^3, 0, 1+z+z^3, 1+z+z^2, 1, 0, 0, 0, 0, 1, 0, 1+z^2, 1+z, z^3, 1+z+z^2+z^3, 1, 1+z^2, 1+z+z^2, z, 1+z+z^2, z+z^2+z^3, 0, 1+z+z^3, 1, 0, 0, 0, 0, 0, 1, 1+z^2, 1+z^2, 1+z, z^3, 1+z+z^2+z^3, 1, 1+z+z^3, 1+z+z^2, z, 1+z+z^2, z+z^2+z^3, 0, 1]);
C_4_19_7 := LinearCode(G_4_19_7);
assert Dimension(C_4_19_7) eq 6;
assert C_4_19_7 subset HermitianDual(C_4_19_7);
d_dual_4_19_7 := MinimumWeight(HermitianDual(C_4_19_7));
assert d_dual_4_19_7 ge 6;
Q_4_19_7 := QuantumCode(C_4_19_7);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_4_19_7), Dimension(Q_4_19_7), 6, 4, d_dual_4_19_7;

// --- [[20, 10, 5]]_4 (CodeTables.de was 4..6) ---
G_4_20_10 := Matrix(F16, 5, 20, [1, 0, 0, 0, 0, 1+z^3, z+z^3, 1+z+z^3, z+z^2, z^2, z+z^2, 1+z+z^2, 0, z^2, 1+z^2+z^3, 1+z^2, 1, 1+z^3, z^3, z+z^3, 0, 1, 0, 0, 0, z^2, 1+z^3, z+z^3, 1+z+z^3, z+z^2, 1+z^2+z^3, z+z^2, 1+z+z^2, 0, z^2, z+z^3, 1+z^2, 1, 1+z^3, z^3, 0, 0, 1, 0, 0, z+z^2, z^2, 1+z^3, z+z^3, 1+z+z^3, z^2, 1+z^2+z^3, z+z^2, 1+z+z^2, 0, z^3, z+z^3, 1+z^2, 1, 1+z^3, 0, 0, 0, 1, 0, 1+z+z^3, z+z^2, z^2, 1+z^3, z+z^3, 0, z^2, 1+z^2+z^3, z+z^2, 1+z+z^2, 1+z^3, z^3, z+z^3, 1+z^2, 1, 0, 0, 0, 0, 1, z+z^3, 1+z+z^3, z+z^2, z^2, 1+z^3, 1+z+z^2, 0, z^2, 1+z^2+z^3, z+z^2, 1, 1+z^3, z^3, z+z^3, 1+z^2]);
C_4_20_10 := LinearCode(G_4_20_10);
assert Dimension(C_4_20_10) eq 5;
assert C_4_20_10 subset HermitianDual(C_4_20_10);
d_dual_4_20_10 := MinimumWeight(HermitianDual(C_4_20_10));
assert d_dual_4_20_10 ge 5;
Q_4_20_10 := QuantumCode(C_4_20_10);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_4_20_10), Dimension(Q_4_20_10), 5, 4, d_dual_4_20_10;

// --- [[25, 15, 5]]_4 (CodeTables.de was 4..5) ---
G_4_25_15 := Matrix(F16, 5, 25, [1, 0, 0, 0, 0, 1, 1+z^2+z^3, z^2+z^3, 1+z+z^3, 1+z+z^2+z^3, 1+z+z^3, 1+z^3, 1+z, 1+z+z^2+z^3, z^3, 1+z^3, z, 1+z+z^2+z^3, 1+z, 1, 1+z^2+z^3, z+z^3, 1+z^2+z^3, 1+z^2, z, 0, 1, 0, 0, 0, 1+z+z^2+z^3, 1, 1+z^2+z^3, z^2+z^3, 1+z+z^3, z^3, 1+z+z^3, 1+z^3, 1+z, 1+z+z^2+z^3, 1, 1+z^3, z, 1+z+z^2+z^3, 1+z, z, 1+z^2+z^3, z+z^3, 1+z^2+z^3, 1+z^2, 0, 0, 1, 0, 0, 1+z+z^3, 1+z+z^2+z^3, 1, 1+z^2+z^3, z^2+z^3, 1+z+z^2+z^3, z^3, 1+z+z^3, 1+z^3, 1+z, 1+z, 1, 1+z^3, z, 1+z+z^2+z^3, 1+z^2, z, 1+z^2+z^3, z+z^3, 1+z^2+z^3, 0, 0, 0, 1, 0, z^2+z^3, 1+z+z^3, 1+z+z^2+z^3, 1, 1+z^2+z^3, 1+z, 1+z+z^2+z^3, z^3, 1+z+z^3, 1+z^3, 1+z+z^2+z^3, 1+z, 1, 1+z^3, z, 1+z^2+z^3, 1+z^2, z, 1+z^2+z^3, z+z^3, 0, 0, 0, 0, 1, 1+z^2+z^3, z^2+z^3, 1+z+z^3, 1+z+z^2+z^3, 1, 1+z^3, 1+z, 1+z+z^2+z^3, z^3, 1+z+z^3, z, 1+z+z^2+z^3, 1+z, 1, 1+z^3, z+z^3, 1+z^2+z^3, 1+z^2, z, 1+z^2+z^3]);
C_4_25_15 := LinearCode(G_4_25_15);
assert Dimension(C_4_25_15) eq 5;
assert C_4_25_15 subset HermitianDual(C_4_25_15);
d_dual_4_25_15 := MinimumWeight(HermitianDual(C_4_25_15));
assert d_dual_4_25_15 ge 5;
Q_4_25_15 := QuantumCode(C_4_25_15);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_4_25_15), Dimension(Q_4_25_15), 5, 4, d_dual_4_25_15;

// --- [[26, 16, 5]]_4 (CodeTables.de was 4..5) ---
G_4_26_16 := Matrix(F16, 5, 26, [1, 0, 0, 0, 0, z+z^2, 1+z^2+z^3, 1, z+z^2+z^3, 0, 1+z, z^3, z^2, z+z^2+z^3, z^2, 1+z+z^2, z^3, 1+z+z^2, 1+z, z^2, 1+z^2+z^3, 1+z, 1+z+z^2, 1, 0, 1, 0, 1, 0, 0, 0, 0, z+z^2, 1+z^2+z^3, 1, z+z^2+z^3, z^2, 1+z, z^3, z^2, z+z^2+z^3, z^2, 1+z+z^2, z^3, 1+z+z^2, 1+z, 0, 1+z^2+z^3, 1+z, 1+z+z^2, 1, 1, 0, 0, 1, 0, 0, z+z^2+z^3, 0, z+z^2, 1+z^2+z^3, 1, z+z^2+z^3, z^2, 1+z, z^3, z^2, 1+z, z^2, 1+z+z^2, z^3, 1+z+z^2, 1, 0, 1+z^2+z^3, 1+z, 1+z+z^2, 1, 0, 0, 0, 1, 0, 1, z+z^2+z^3, 0, z+z^2, 1+z^2+z^3, z^2, z+z^2+z^3, z^2, 1+z, z^3, 1+z+z^2, 1+z, z^2, 1+z+z^2, z^3, 1+z+z^2, 1, 0, 1+z^2+z^3, 1+z, 1, 0, 0, 0, 0, 1, 1+z^2+z^3, 1, z+z^2+z^3, 0, z+z^2, z^3, z^2, z+z^2+z^3, z^2, 1+z, z^3, 1+z+z^2, 1+z, z^2, 1+z+z^2, 1+z, 1+z+z^2, 1, 0, 1+z^2+z^3, 1]);
C_4_26_16 := LinearCode(G_4_26_16);
assert Dimension(C_4_26_16) eq 5;
assert C_4_26_16 subset HermitianDual(C_4_26_16);
d_dual_4_26_16 := MinimumWeight(HermitianDual(C_4_26_16));
assert d_dual_4_26_16 ge 5;
Q_4_26_16 := QuantumCode(C_4_26_16);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_4_26_16), Dimension(Q_4_26_16), 5, 4, d_dual_4_26_16;

// --- [[19, 7, 6]]_5 (CodeTables.de was 5..7) ---
G_5_19_7 := Matrix(F25, 6, 19, [1, 0, 0, 0, 0, 0, 4+t, 1, 4, 2, 2+2*t, 2+2*t, 1+4*t, 2+4*t, t, 3+4*t, 3+2*t, 1+t, 1, 0, 1, 0, 0, 0, 0, 2+2*t, 4+t, 1, 4, 2, 2+2*t, 1+t, 1+4*t, 2+4*t, t, 3+4*t, 3+2*t, 1, 0, 0, 1, 0, 0, 0, 2+2*t, 2+2*t, 4+t, 1, 4, 2, 3+2*t, 1+t, 1+4*t, 2+4*t, t, 3+4*t, 1, 0, 0, 0, 1, 0, 0, 2, 2+2*t, 2+2*t, 4+t, 1, 4, 3+4*t, 3+2*t, 1+t, 1+4*t, 2+4*t, t, 1, 0, 0, 0, 0, 1, 0, 4, 2, 2+2*t, 2+2*t, 4+t, 1, t, 3+4*t, 3+2*t, 1+t, 1+4*t, 2+4*t, 1, 0, 0, 0, 0, 0, 1, 1, 4, 2, 2+2*t, 2+2*t, 4+t, 2+4*t, t, 3+4*t, 3+2*t, 1+t, 1+4*t, 1]);
C_5_19_7 := LinearCode(G_5_19_7);
assert Dimension(C_5_19_7) eq 6;
assert C_5_19_7 subset HermitianDual(C_5_19_7);
d_dual_5_19_7 := MinimumWeight(HermitianDual(C_5_19_7));
assert d_dual_5_19_7 ge 6;
Q_5_19_7 := QuantumCode(C_5_19_7);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_5_19_7), Dimension(Q_5_19_7), 6, 5, d_dual_5_19_7;

// --- [[21, 7, 7]]_7 (CodeTables.de was 6..8) ---
G_7_21_7 := Matrix(F49, 7, 21, [1, 0, 0, 0, 0, 0, 0, 3+5*t, 1+2*t, 2, 1+3*t, 1+3*t, 0, 6+3*t, 2+2*t, 4+5*t, 2+t, 4*t, 3+3*t, 4+3*t, 2+5*t, 0, 1, 0, 0, 0, 0, 0, 6+3*t, 3+5*t, 1+2*t, 2, 1+3*t, 1+3*t, 0, 2+5*t, 2+2*t, 4+5*t, 2+t, 4*t, 3+3*t, 4+3*t, 0, 0, 1, 0, 0, 0, 0, 0, 6+3*t, 3+5*t, 1+2*t, 2, 1+3*t, 1+3*t, 4+3*t, 2+5*t, 2+2*t, 4+5*t, 2+t, 4*t, 3+3*t, 0, 0, 0, 1, 0, 0, 0, 1+3*t, 0, 6+3*t, 3+5*t, 1+2*t, 2, 1+3*t, 3+3*t, 4+3*t, 2+5*t, 2+2*t, 4+5*t, 2+t, 4*t, 0, 0, 0, 0, 1, 0, 0, 1+3*t, 1+3*t, 0, 6+3*t, 3+5*t, 1+2*t, 2, 4*t, 3+3*t, 4+3*t, 2+5*t, 2+2*t, 4+5*t, 2+t, 0, 0, 0, 0, 0, 1, 0, 2, 1+3*t, 1+3*t, 0, 6+3*t, 3+5*t, 1+2*t, 2+t, 4*t, 3+3*t, 4+3*t, 2+5*t, 2+2*t, 4+5*t, 0, 0, 0, 0, 0, 0, 1, 1+2*t, 2, 1+3*t, 1+3*t, 0, 6+3*t, 3+5*t, 4+5*t, 2+t, 4*t, 3+3*t, 4+3*t, 2+5*t, 2+2*t]);
C_7_21_7 := LinearCode(G_7_21_7);
assert Dimension(C_7_21_7) eq 7;
assert C_7_21_7 subset HermitianDual(C_7_21_7);
d_dual_7_21_7 := MinimumWeight(HermitianDual(C_7_21_7));
assert d_dual_7_21_7 ge 7;
Q_7_21_7 := QuantumCode(C_7_21_7);
printf "VERIFIED [[%o, %o, %o]]_%o (pure, C^perp_H min weight = %o)\n", Length(Q_7_21_7), Dimension(Q_7_21_7), 7, 7, d_dual_7_21_7;

print "ALL 10 QUANTUM CODES VERIFIED SUCCESSFULLY!";
quit;
