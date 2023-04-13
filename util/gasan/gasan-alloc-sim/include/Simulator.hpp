#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <ostream>

namespace gasan::alloc_sim {

/**
 * \brief Statistics for the simulator
 * \see Simulator
 */
struct Stats {

  size_t max_us;      //!< Maximum number of bytes allocated by the user
  size_t max_rq;      //!< Maximum amount of memory used
  size_t max_shad_rq; //!< Maximum amount of shadow memory required
  size_t max_tot_rq;  //!< Sum of all memory required, including shadow memory

  double max_ov;      //!< Overhead with fragmentation and redzones
  double max_shad_ov; //!< Overhead of shadow memory
  double max_tot_ov;  //!< Total overhead

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
   * \brief Information for an allocation
   *
   * This is contained in the Region housing the allocation. The redzones are
   * not included since they are free regions.
   */
  struct AllocationInfo {
    uint64_t tag;     //!< The tag associated with this allocation
    size_t user_size; //!< Allocation size requested by the user

    /**
     * \return The size of the redzone needed on each side of this allocation
     */
    size_t redzoneSize() const;
  };

  /**
   * \brief A region of memory
   */
  struct Region {
    size_t start; //!< First address of the region
    size_t end;   //!< Just past the last address of the region

    /**
     * \brief Whether this region is allocated, and the information for it
     */
    std::optional<AllocationInfo> alloc_info;

    /**
     * \brief Compares to regions if they don't overlap
     *
     * Also checks for equality. That's the only case where Regions aren't
     * unordered when they overlap.
     *
     * \return Whether one regions is entirely to the left or right of the
     * other
     */
    std::partial_ordering operator<=>(const Region &that) const;

    /**
     * \return The difference between the start and the end
     */
    size_t size() const;
    /**
     * \brief Check if this is logically the "infinite" region
     *
     * Internally, this is modelled as having the end be the maximum possible
     * value for `size_t`. We should never need that much memory. There is
     * logically one infinite region at the end of the list of regions.
     *
     * \return Whether the end stretches to the maximum value
     */
    bool isInfinite() const;

    /**
     * \brief Check if we have an allocation in this region
     */
    bool isAllocated() const;
  };

  /**
   * \brief List of all the regions we know about
   *
   * Instead of storing the free and allocated regions separately, we just keep
   * them in a single list. This makes it easier to handle redzones, which must
   * be in free regions.
   */
  std::list<Region> region_list_;
};

std::ostream &operator<<(std::ostream &os, const Stats &s);

} // namespace gasan::alloc_sim
