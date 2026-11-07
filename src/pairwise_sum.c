#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_N 1000000000
#define ELEMENT 1
#define TRACE_MAX 64
#define EXCHANGE_CHUNK (1 << 22)

typedef struct {
    int peer;
    int dest_lo;
    int src_lo;
    int count;
    int *buf;
    int is_self;
} Piece;

static void die(int rank, const char *message) {
    fprintf(stderr, "rank %d: %s\n", rank, message);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static int *alloc_ints(int rank, int count, const char *what) {
    if (count < 0) {
        die(rank, what);
    }
    if (count == 0) {
        return NULL;
    }
    if ((size_t)count > SIZE_MAX / sizeof(int)) {
        die(rank, what);
    }
    int *buffer = malloc((size_t)count * sizeof(int));
    if (buffer == NULL) {
        die(rank, what);
    }
    return buffer;
}

static int owner_of(const int *displs, const int *counts, int size, int global) {
    int lo = 0;
    int hi = size - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int start = displs[mid];
        int end = start + counts[mid];
        if (global < start) {
            hi = mid - 1;
        } else if (global >= end) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}

static void build_distribution(int n, int size, int *counts, int *displs) {
    int base = n / size;
    int rem = n % size;
    int offset = 0;
    for (int rank = 0; rank < size; rank++) {
        counts[rank] = base + (rank < rem ? 1 : 0);
        displs[rank] = offset;
        offset += counts[rank];
    }
}

static void usage(void) {
    fprintf(stderr,
            "usage: pairwise_sum [N] [--demo]\n"
            "  N      array length (default %d), each element is %d\n"
            "  --demo elements are 1..N; waves are printed when N <= %d\n",
            DEFAULT_N, ELEMENT, TRACE_MAX);
}

static void parse_args(int argc, char **argv, int rank, int *n, int *demo) {
    *n = DEFAULT_N;
    *demo = 0;
    if (argc > 3) {
        if (rank == 0) {
            usage();
        }
        die(rank, "too many arguments");
    }
    if (argc >= 2) {
        char *end = NULL;
        errno = 0;
        long long value = strtoll(argv[1], &end, 10);
        if (errno != 0 || end == argv[1] || *end != '\0' || value < 0 || value > INT_MAX) {
            if (rank == 0) {
                usage();
            }
            die(rank, "N must be an integer in [0, 2147483647]");
        }
        *n = (int)value;
    }
    if (argc == 3) {
        if (strcmp(argv[2], "--demo") != 0) {
            if (rank == 0) {
                usage();
            }
            die(rank, "unknown option");
        }
        *demo = 1;
    }
    if (*demo) {
        long long triangular = (long long)(*n) * (*n + 1) / 2;
        if (triangular > INT_MAX) {
            die(rank, "demo N is too large for a 32-bit in-place sum");
        }
    }
}

static int plan_sends(int rank, int size, int local_displ, int local_count, int length,
                      int win_lo, int win_hi, const int *displs, const int *counts, Piece *pieces) {
    int src_begin = local_displ;
    int src_end = local_displ + local_count;
    int src_limit_lo = length - win_hi;
    int src_limit_hi = length - win_lo;
    if (src_begin < src_limit_lo) {
        src_begin = src_limit_lo;
    }
    if (src_end > src_limit_hi) {
        src_end = src_limit_hi;
    }
    if (src_begin >= src_end) {
        return 0;
    }

    int n_pieces = 0;
    int cursor = src_begin;
    while (cursor < src_end) {
        if (n_pieces >= size) {
            die(rank, "send split overflow");
        }
        int dest = length - 1 - cursor;
        int dest_rank = owner_of(displs, counts, size, dest);
        if (dest_rank < 0) {
            die(rank, "destination rank not found");
        }

        int dest_lo_bound = displs[dest_rank];
        int max_source = length - 1 - dest_lo_bound;
        int piece_hi = src_end;
        if (piece_hi > max_source + 1) {
            piece_hi = max_source + 1;
        }
        if (piece_hi <= cursor) {
            die(rank, "send split did not advance");
        }

        int count = piece_hi - cursor;
        int dest_lo = length - piece_hi;
        int dest_hi = dest_lo + count;
        int owned_hi = dest_lo_bound + counts[dest_rank];
        if (dest_lo < dest_lo_bound || dest_hi > owned_hi) {
            die(rank, "send piece left the destination rank");
        }

        Piece *piece = &pieces[n_pieces++];
        piece->peer = dest_rank;
        piece->dest_lo = dest_lo;
        piece->src_lo = cursor;
        piece->count = count;
        piece->buf = NULL;
        piece->is_self = dest_rank == rank;
        cursor = piece_hi;
    }
    return n_pieces;
}

static int plan_recvs(int rank, int size, int local_displ, int local_count, int length,
                      int win_lo, int win_hi, const int *displs, const int *counts, Piece *pieces) {
    int dest_begin = local_displ;
    int dest_end = local_displ + local_count;
    if (dest_begin < win_lo) {
        dest_begin = win_lo;
    }
    if (dest_end > win_hi) {
        dest_end = win_hi;
    }
    if (dest_begin >= dest_end) {
        return 0;
    }

    int n_pieces = 0;
    int cursor = dest_begin;
    while (cursor < dest_end) {
        if (n_pieces >= size) {
            die(rank, "recv split overflow");
        }
        int src = length - 1 - cursor;
        int src_rank = owner_of(displs, counts, size, src);
        if (src_rank < 0) {
            die(rank, "source rank not found");
        }

        int src_lo_bound = displs[src_rank];
        int max_dest = length - 1 - src_lo_bound;
        int piece_hi = dest_end;
        if (piece_hi > max_dest + 1) {
            piece_hi = max_dest + 1;
        }
        if (piece_hi <= cursor) {
            die(rank, "recv split did not advance");
        }

        if (src_rank == rank) {
            cursor = piece_hi;
            continue;
        }

        int count = piece_hi - cursor;
        int src_lo = length - piece_hi;
        int src_hi = length - cursor;
        int owned_hi = src_lo_bound + counts[src_rank];
        if (src_lo < src_lo_bound || src_hi > owned_hi) {
            die(rank, "recv piece left the source rank");
        }

        Piece *piece = &pieces[n_pieces++];
        piece->peer = src_rank;
        piece->dest_lo = cursor;
        piece->src_lo = src_lo;
        piece->count = count;
        piece->buf = NULL;
        piece->is_self = 0;
        cursor = piece_hi;
    }
    return n_pieces;
}

static void add_local_piece(int *data, int local_displ, const Piece *piece) {
    int *dest = data + (piece->dest_lo - local_displ);
    for (int i = 0; i < piece->count; i++) {
        int src_global = piece->src_lo + piece->count - 1 - i;
        dest[i] += data[src_global - local_displ];
    }
}

static void pack_reversed(const int *data, int local_displ, const Piece *piece) {
    for (int i = 0; i < piece->count; i++) {
        int src_global = piece->src_lo + piece->count - 1 - i;
        piece->buf[i] = data[src_global - local_displ];
    }
}

static void fold_window(int rank, int size, int local_displ, int local_count, int *data,
                        int length, int win_lo, int win_hi, int tag, const int *displs,
                        const int *counts, Piece *sends, Piece *recvs, MPI_Request *requests) {
    int n_sends = plan_sends(rank, size, local_displ, local_count, length, win_lo, win_hi,
                             displs, counts, sends);
    int n_recvs = plan_recvs(rank, size, local_displ, local_count, length, win_lo, win_hi,
                             displs, counts, recvs);
    int n_requests = 0;

    for (int i = 0; i < n_recvs; i++) {
        recvs[i].buf = alloc_ints(rank, recvs[i].count, "out of memory for receive");
        MPI_Irecv(recvs[i].buf, recvs[i].count, MPI_INT, recvs[i].peer, tag,
                  MPI_COMM_WORLD, &requests[n_requests]);
        n_requests++;
    }

    for (int i = 0; i < n_sends; i++) {
        if (sends[i].is_self) {
            continue;
        }
        sends[i].buf = alloc_ints(rank, sends[i].count, "out of memory for send");
        pack_reversed(data, local_displ, &sends[i]);
        MPI_Isend(sends[i].buf, sends[i].count, MPI_INT, sends[i].peer, tag,
                  MPI_COMM_WORLD, &requests[n_requests]);
        n_requests++;
    }

    for (int i = 0; i < n_sends; i++) {
        if (sends[i].is_self) {
            add_local_piece(data, local_displ, &sends[i]);
        }
    }

    if (n_requests > 0) {
        MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < n_recvs; i++) {
        int *dest = data + (recvs[i].dest_lo - local_displ);
        for (int k = 0; k < recvs[i].count; k++) {
            dest[k] += recvs[i].buf[k];
        }
        free(recvs[i].buf);
        recvs[i].buf = NULL;
    }
    for (int i = 0; i < n_sends; i++) {
        free(sends[i].buf);
        sends[i].buf = NULL;
    }
}

static void fold_wave(int rank, int size, int local_displ, int local_count, int *data,
                      int length, int wave, const int *displs, const int *counts,
                      Piece *sends, Piece *recvs, MPI_Request *requests) {
    int pairs = length / 2;
    int chunk_index = 0;
    for (int win_lo = 0; win_lo < pairs; win_lo += EXCHANGE_CHUNK) {
        int win_hi = win_lo + EXCHANGE_CHUNK;
        if (win_hi > pairs) {
            win_hi = pairs;
        }
        fold_window(rank, size, local_displ, local_count, data, length, win_lo, win_hi,
                    wave * 100000 + chunk_index, displs, counts, sends, recvs, requests);
        chunk_index++;
    }
}

static void print_state(int rank, int n, int local_count, const int *data,
                        const int *counts, const int *displs, int wave, int active_length) {
    int dummy = 0;
    const int *send = local_count > 0 ? data : &dummy;
    int *all = NULL;
    if (rank == 0) {
        all = alloc_ints(rank, n, "out of memory for wave trace");
    }
    MPI_Gatherv((void *)send, local_count, MPI_INT,
                all, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("wave %d length %d:", wave, active_length);
        for (int i = 0; i < n; i++) {
            printf(" %d", all[i]);
        }
        printf("\n");
        fflush(stdout);
        free(all);
    }
}

static double pairwise_sum(int rank, int size, int n, int local_count, int *data,
                           const int *counts, const int *displs, int trace) {
    Piece *sends = malloc((size_t)size * sizeof(Piece));
    Piece *recvs = malloc((size_t)size * sizeof(Piece));
    MPI_Request *requests = malloc((size_t)size * 2 * sizeof(MPI_Request));
    if (sends == NULL || recvs == NULL || requests == NULL) {
        die(rank, "out of memory for wave plan");
    }

    int local_displ = displs[rank];
    if (trace) {
        print_state(rank, n, local_count, data, counts, displs, 0, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double started = MPI_Wtime();

    int length = n;
    int wave = 0;
    while (length > 1) {
        wave++;
        fold_wave(rank, size, local_displ, local_count, data, length, wave,
                  displs, counts, sends, recvs, requests);
        MPI_Barrier(MPI_COMM_WORLD);
        length -= length / 2;
        if (trace) {
            print_state(rank, n, local_count, data, counts, displs, wave, length);
        }
    }

    double seconds = MPI_Wtime() - started;
    free(sends);
    free(recvs);
    free(requests);
    return seconds;
}

int main(int argc, char **argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    setlocale(LC_NUMERIC, "C");

    int n = 0;
    int demo = 0;
    parse_args(argc, argv, rank, &n, &demo);

    int *counts = malloc((size_t)size * sizeof(int));
    int *displs = malloc((size_t)size * sizeof(int));
    if (counts == NULL || displs == NULL) {
        die(rank, "out of memory for distribution");
    }
    build_distribution(n, size, counts, displs);

    int local_count = counts[rank];
    int *data = NULL;
    if (local_count > 0) {
        data = alloc_ints(rank, local_count, "out of memory for the local block");
        int origin = displs[rank];
        for (int i = 0; i < local_count; i++) {
            data[i] = demo ? origin + i + 1 : ELEMENT;
        }
    }

    int trace = demo && n > 0 && n <= TRACE_MAX;
    double seconds = pairwise_sum(rank, size, n, local_count, data, counts, displs, trace);

    int sum = 0;
    if (rank == 0 && n > 0) {
        sum = data[0];
    }

    int ok = 1;
    if (rank == 0) {
        long long expected = demo ? (long long)n * (n + 1) / 2 : (long long)n;
        double time_ms = seconds * 1000.0;
        printf("sum=%d, timeMs=%.2f\n", sum, time_ms);
        fflush(stdout);
        if ((long long)sum != expected) {
            fprintf(stderr, "mismatch: sum=%d expected=%lld\n", sum, expected);
            fflush(stderr);
            ok = 0;
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    free(data);
    free(counts);
    free(displs);
    MPI_Finalize();
    return ok ? 0 : 1;
}
