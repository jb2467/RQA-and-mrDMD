from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def count_line_lengths(binary_matrix, direction='diagonal'):
  N = binary_matrix.shape[0]
  lengths = []

  if direction == 'diagonal':
    for k in range(-N+1, N): 
      diag = np.diag(binary_matrix, k)
      count = 0
      for val in diag:
        if val:
          count += 1
        elif count > 0:
          lengths.append(count)
          count = 0
      if count > 0:
        lengths.append(count)

  elif direction == 'vertical':
    for col in range(binary_matrix.shape[1]):
      count = 0
      for row in range(binary_matrix.shape[0]):
        if binary_matrix[row, col]:
          count += 1
        elif count > 0:
          lengths.append(count)
          count = 0
      if count > 0:
        lengths.append(count)

  return Counter(lengths)

def rqa_metrics(recurrence_matrix, l_min=2, v_min=2):
  R = np.array(recurrence_matrix, dtype=int)
  total_rec_points = R.sum()
  diag_lines = count_line_lengths(R, direction='diagonal')
  vert_lines = count_line_lengths(R, direction='vertical')
    # DET
  det_numer = sum(l * count for l, count in diag_lines.items() if l >= l_min)
  DET = det_numer / total_rec_points if total_rec_points > 0 else 0
    # LAM
  lam_numer = sum(v * count for v, count in vert_lines.items() if v >= v_min)
  LAM = lam_numer / total_rec_points if total_rec_points > 0 else 0
    # ENTR
  diag_filtered = {l: c for l, c in diag_lines.items() if l >= l_min}
  total_lines = sum(diag_filtered.values())
  if total_lines > 0:
    probs = np.array([c / total_lines for c in diag_filtered.values()])
    ENTR = -np.sum(probs * np.log(probs))
    length = len(probs)
    ENTR = ENTR / np.log(length)
  else:
    ENTR = 0
  return {
        'DET': DET,
        'LAM': LAM,
        'ENTR': ENTR,
    }


def downsample(dynamics, q):
    all_vectors = dynamics.copy()
    n_samples_target = (all_vectors.shape[0]//q)
    indices = np.linspace(0, all_vectors.shape[0]-1, n_samples_target).astype(int)
    downsampled = all_vectors[indices, :]
    all_vectors = downsampled
    return all_vectors


def correlation_distance(all_vectors):
    all_vectors = np.abs(all_vectors)
    distance_matrix = cdist(all_vectors, all_vectors, metric='correlation')
    return distance_matrix


def manhattan_distance(all_vectors):
    all_vectors = np.abs(all_vectors)
    distance_matrix = cdist(all_vectors, all_vectors, metric='cityblock')
    return distance_matrix

def cosine_distance(all_vectors):
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    normalized = all_vectors / (norms + 1e-10)  
    similarity = normalized @ normalized.T
    distance_matrix = 1.0 - similarity
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0, 2)

    return distance_matrix


def euclidean_distance(all_vectors):
    sq_norm = np.sum(np.abs(all_vectors)**2, axis=1)
    gram = all_vectors @ np.conj(all_vectors).T

    real_gram = gram.real

    sq_distances = (sq_norm[:, np.newaxis] + sq_norm[np.newaxis, :] - 2 * real_gram)

    np.fill_diagonal(sq_distances, 0)

    sq_distances = np.maximum(sq_distances, 0)

    distance_matrix = np.sqrt(sq_distances)
    return distance_matrix 


def distance(all_vectors, metric='euclidean'):
    if metric == 'euclidean':
        return euclidean_distance(all_vectors)
    elif metric == 'cosine':
        return cosine_distance(all_vectors)
    elif metric == 'manhattan':
        return manhattan_distance(all_vectors)
    elif metric == 'correlation':
        return correlation_distance(all_vectors)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")




# Test this more
def RecurrencePlot(all_vectors, percent, metric = 'euclidean', q=1, v_min=2 , l_min =2, globalEpsilon = True, Title = '', lines= False):
    temp = all_vectors.copy()
    temp = downsample(temp,q)
    distance_matrix = distance(temp, metric = metric)
    metrics = 0 
    if globalEpsilon:
        epsilon = np.percentile(distance_matrix,percent )
        print(f"epsilon = {epsilon:.3f}")
        R = (distance_matrix <= epsilon)
        np.fill_diagonal(R, 0)
        metrics = rqa_metrics(R, l_min=l_min, v_min=v_min)
        np.fill_diagonal(R, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(R, origin='lower', cmap='binary')
        if lines:       
            plt.axvline(165, color='red', linestyle='dotted', label='1st Regime change')
            plt.axvline(286, color='red', linestyle='dotted', label='2nd Regime change')
            plt.axvline(376, color='red', linestyle='dotted', label='3rd Regime change')
            plt.axvline(501, color='red', linestyle='dotted', label='4th Regime change')
            plt.axvline(751, color='red', linestyle='dotted', label='5th Regime change')
        plt.title(f"{Title} \nRecurrence Plot (ε = {epsilon:.3f}), RR: {percent/100} \nwith DET:{metrics['DET']:.3f}, LAM:{metrics['LAM']:.3f}, ENTR:{metrics['ENTR']:.3f}", fontsize=16)
        plt.xlabel('Time index j')
        plt.ylabel('Time index i')
        plt.tight_layout()
        plt.show()

    else:
        N = distance_matrix.shape[0]
        k = int((percent/100)*N)
        R = np.zeros((N, N), dtype=int)
        for i in range(N):
            neighbors = np.argsort(distance_matrix[i])[:k+1]
            neighbors = neighbors[neighbors != i][:k]
            R[i, neighbors] = 1
            R[neighbors, i] = 1
        metrics = rqa_metrics(R, l_min=l_min, v_min=v_min)
        np.fill_diagonal(R, 1)
        plt.figure(figsize=(10,10))
        plt.imshow(R, origin='lower', cmap='binary')
        if lines:
            '''
            #plt.axvline(165, color='red', linestyle='dotted', label='1st Regime change')
            plt.axvline(286-200, color='red', linestyle='dotted', label='2nd Regime change')
            plt.axvline(376-200, color='red', linestyle='dotted', label='3rd Regime change')
            plt.axvline(501-200, color='red', linestyle='dotted', label='4th Regime change')
            plt.axvline(751-200, color='red', linestyle='dotted', label='5th Regime change')
            
            '''
            plt.axvline(165, color='red', linestyle='dotted', label='1st Regime change')
            plt.axvline(286, color='red', linestyle='dotted', label='2nd Regime change')
            plt.axvline(376, color='red', linestyle='dotted', label='3rd Regime change')
            plt.axvline(501, color='red', linestyle='dotted', label='4th Regime change')
            plt.axvline(751, color='red', linestyle='dotted', label='5th Regime change')
        plt.xlabel('time index')
        plt.ylabel('time index')
        plt.title(f"{Title} \nRecurrence Plot {percent}% Nearest Neighbors \nwith DET:{metrics['DET']:.3f}, LAM:{metrics['LAM']:.3f}, ENTR:{metrics['ENTR']:.3f}" , fontsize=16)
        plt.show()
    return metrics,distance_matrix   
      
    
    
    

    
    
def sliding_window_rqa(all_vectors, percent, metric='euclidean', q=1, window_size=100, step=1, l_min=2, v_min=2, globalEpsilon=True):
    temp = all_vectors.copy()
    if q > 1:
        temp = downsample(temp, q)
    dist_mat = distance(temp, metric=metric)

    N = dist_mat.shape[0]
    half_w = window_size // 2
    times = []
    det_ts, lam_ts, entr_ts = [], [], []

    if globalEpsilon:
        eps = np.percentile(dist_mat, percent)

    for center in range(half_w, N-half_w, step):
        i0, i1 = center-half_w, center+half_w
        subDist = dist_mat[i0:i1, i0:i1]
        if globalEpsilon:
            R = (subDist <= eps).astype(int)
        else:
            k = int((percent/100)*subDist.shape[0])
            R = np.zeros_like(subDist, dtype=int)
            for i in range(subDist.shape[0]):
                nbrs = np.argsort(subDist[i])[:k+1]
                nbrs = nbrs[nbrs != i][:k]
                R[i, nbrs] = 1
                R[nbrs, i] = 1
        np.fill_diagonal(R, 0)

        metrics = rqa_metrics(R, l_min=l_min, v_min=v_min)
        det_ts.append(metrics['DET'])
        lam_ts.append(metrics['LAM'])
        entr_ts.append(metrics['ENTR'])
        times.append(center)

    return np.array(times), det_ts, lam_ts, entr_ts
