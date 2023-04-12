#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <set>
#include <unordered_map>

namespace gasan::alloc_sim {

struct Stats {

  /**
   * \brief Pretty-print the statistics
   */
  friend std::ostream &operator<<(std::ostream &os, const Stats &s);
};

class Simulator {
public:
  /**
   * \brief Initialize the simulator
   */
  Simulator();

  /**
   * \defgroup calls Functions to simulate allocation calls
   * @{
   */
  void cudaMalloc(uint64_t tag, size_t sz);
  void cudaFree(uint64_t tag);
  /**@} */

  /**
   * \brief Get the statistics accumulated so far
   */
  Stats getStats() const;

private:
  /**
   * \brief The statistics we've accumulated so far
   */
  Stats stats_;

  /**
   * \brief Update the statistics
   *
   * Should be called at the end of every simulated function.
   */
  void updateStats();

  /**
   * \brief Allocation granularity
   *
   * Computed from the scale parameter
   */
  size_t granularity_;

  /**
   * \brief Represents a region of memory that can back an allocation
   *
   * This is mainly used by the free list, but it is also used by allocations so
   * they can be freed. These have a partial order, but elements in the free
   * list should be totally ordered.
   */
  struct Region {
    size_t start;              //!< Starting internal address of the region
    std::optional<size_t> end; //!< Ending internal address, or infinity

    std::partial_ordering operator<=>(const Region &that) const;

    /**
     * \brief Computes the size of this region
     *
     * Returns `end - start`, or `nullopt` if the region is infinite.
     */
    std::optional<size_t> size() const;
    /**
     * \brief Checks if the region is infinite
     */
    bool isInfinite() const;
  };

  /**
   * \brief Metadata for allocations
   *
   * Used for calculating statistics and to remember how to free allocations.
   */
  struct AllocationInfo {
    Region region; //!< Region backing this allocation
    size_t size;   //!< Size requested by this allocation
  };

  /**
   * \brief Set of regions we can allocate from
   */
  std::set<Region> free_list_;

  /**
   * \brief Set of currently active allocations
   */
  std::unordered_map<size_t, AllocationInfo> alloc_list_;
};

std::ostream &operator<<(std::ostream &os, const Stats &s);

} // namespace gasan::alloc_sim
