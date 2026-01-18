package net.faulj.spaces;

import net.faulj.vector.Vector;

/**
 * Computes and represents the orthogonal complement of a subspace.
 * <p>
 * For a subspace \( W \) of an inner product space \( V \), the orthogonal complement
 * \( W^\perp \) (pronounced "W perp") is the set of all vectors in \( V \) that are
 * orthogonal to every vector in \( W \).
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * \( W^\perp = \{ x \in V : \langle x, w \rangle = 0 \text{ for all } w \in W \} \)
 * </p>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Direct Sum:</b> \( V = W \oplus W^\perp \) (for finite dimensional spaces).</li>
 * <li><b>Unique Decomposition:</b> Any \( v \in V \) can be uniquely written as \( v = \hat{y} + z \)
 * where \( \hat{y} \in W \) and \( z \in W^\perp \).</li>
 * <li><b>Dimensions:</b> \( \dim(W) + \dim(W^\perp) = \dim(V) \).</li>
 * <li><b>Double Complement:</b> \( (W^\perp)^\perp = W \).</li>
 * </ul>
 *
 * <h2>Fundamental Subspaces:</h2>
 * <p>
 * In the context of a matrix \( A \):
 * </p>
 * <ul>
 * <li>(Row Space \( A \))\(^\perp\) = Null Space \( A \)</li>
 * <li>(Column Space \( A \))\(^\perp\) = Null Space \( A^T \) (Left Null Space)</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SubspaceBasis
 * @see Projection
 */
public class OrthogonalComplement {
}