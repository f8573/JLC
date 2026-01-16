package net.faulj.core;

import java.util.Arrays;

/**
 * Represents a permutation vector for pivoting operations.
 * Tracks row/column exchanges during factorizations.
 */
public class PermutationVector {
    
    private final int[] perm;
    private int exchangeCount;
    
    public PermutationVector(int size) {
        this.perm = new int[size];
        for (int i = 0; i < size; i++) {
            perm[i] = i;
        }
        this.exchangeCount = 0;
    }
    
    public void exchange(int i, int j) {
        if (i != j) {
            int temp = perm[i];
            perm[i] = perm[j];
            perm[j] = temp;
            exchangeCount++;
        }
    }
    
    public int get(int index) {
        return perm[index];
    }
    
    public int size() {
        return perm.length;
    }
    
    public int getExchangeCount() {
        return exchangeCount;
    }
    
    /**
     * Returns the sign of the permutation: +1 for even, -1 for odd exchanges.
     */
    public int sign() {
        return (exchangeCount % 2 == 0) ? 1 : -1;
    }
    
    public int[] toArray() {
        return Arrays.copyOf(perm, perm.length);
    }
    
    public PermutationVector copy() {
        PermutationVector p = new PermutationVector(perm.length);
        System.arraycopy(this.perm, 0, p.perm, 0, perm.length);
        p.exchangeCount = this.exchangeCount;
        return p;
    }
}
