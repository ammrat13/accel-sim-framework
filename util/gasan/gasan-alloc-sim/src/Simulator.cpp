#include "Simulator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "arguments.hpp"

using namespace gasan::alloc_sim;

// Check type equivalences
// Needed for integer literals
static_assert(std::is_same<size_t, unsigned long>::value);

/**
 * \brief Rounds `v` to the nearest multiple of `m`
 *
 * Only works if both arguments are integers.
 */
#define INTROUNDTO(v, m) ((((v) + (m)-1) / (m)) * (m))

std::ostream &gasan::alloc_sim::operator<<(std::ostream &os, const Stats &s) {
  os << "max_us      " << s.max_us << '\n';
  os << "max_rq      " << s.max_rq << '\n';
  os << "max_shad_rq " << s.max_shad_rq << '\n';
  os << "max_tot_rq  " << s.max_tot_rq << '\n';
  os << "max_shad_ov " << s.max_shad_ov << '\n';
  os << "max_tot_ov  " << s.max_tot_ov << '\n';
  return os;
}

size_t Simulator::AllocationInfo::redzoneSize() const {
  // Retrieve the constants
  size_t B = ARGS.get<size_t>("--redzone-base");
  double L = ARGS.get<double>("--redzone-linear");
  // Return
  return std::max(B, (size_t)std::ceil(L * this->user_size));
}

std::partial_ordering Simulator::Region::operator<=>(const Region &that) const {

  // Assert both regions have their ends after their starts
  assert(this->end >= this->start &&
         "This region's end must be after its start");
  assert(that.end >= that.start && "That region's end must be after its start");

  // If the starts are equal, return equal if the ends are, otherwise
  // incomparable
  if (this->start == that.start)
    return this->end == that.end ? std::partial_ordering::equivalent
                                 : std::partial_ordering::unordered;

  // If this start is to the left, check if this end is also to the left,
  // otherwise incomparable.
  if (this->start < that.start) {
    if (this->end <= that.start)
      return std::partial_ordering::less;
    return std::partial_ordering::unordered;
  }

  // Same as above, but with the roles reversed
  if (this->start > that.start) {
    if (this->start >= that.end)
      return std::partial_ordering::greater;
    return std::partial_ordering::unordered;
  }

  // All cases have been handled
  assert(false && "Starts not totally ordered");
}

size_t Simulator::Region::size() const {
  // Check that the end is after the start, then return
  assert(this->end >= this->start && "Region end must be after start");
  return this->end - this->start;
}

bool Simulator::Region::isInfinite() const {
  return this->end == std::numeric_limits<size_t>::max();
}

bool Simulator::Region::isAllocated() const {
  return this->alloc_info.has_value();
}

Simulator::Simulator()
    : granularity_{1ul << ARGS.get<size_t>("--scale")}, stats_{} {
  // Initialize the freelist with one huge chunk
  this->region_list_.push_back(
      {.start = 0ul, .end = std::numeric_limits<size_t>::max()});
  // Update the statistics
  // We mainly do this for invariant checking
  this->updateStats();
}

Stats Simulator::getStats() const { return this->stats_; }

void Simulator::updateStats() {
  // Do assertions here
  // These take quadratic time, so only do them in debug builds
#ifndef NDEBUG

  // The region list should not be empty
  assert(!this->region_list_.empty() && "Must have at least one region");

  // The last element of the region list should be the infinite region
  assert(this->region_list_.crbegin()->isInfinite() &&
         "Last region must be the infinite region");
  // That should be the only infinite region
  assert(std::all_of(this->region_list_.cbegin(),
                     std::prev(this->region_list_.cend()),
                     [](const auto &r) { return !r.isInfinite(); }) &&
         "Must have only one infinite region");
  // The infinite region should not be allocated
  assert(!this->region_list_.crbegin()->isAllocated() &&
         "The infinite region must not be allocated");

  // Assert that all the elements of the region list are contiguous
  std::accumulate(this->region_list_.cbegin(), this->region_list_.cend(), 0ul,
                  [](size_t p, const auto &r) {
                    assert(r.start == p && "Regions must be contiguous");
                    return r.end;
                  });
  // Assert that all the elements of the region list are sorted
  for (auto i = this->region_list_.cbegin(); i != this->region_list_.cend();
       i++) {
    for (auto j = this->region_list_.cbegin(); j != i; j++) {
      assert(*j < *i && "Region list must be sorted");
    }
  }
#endif

  // Sum all the user sizes
  this->stats_.max_us = std::max(
      this->stats_.max_us,
      std::accumulate(this->region_list_.cbegin(), this->region_list_.cend(),
                      0ul, [](size_t us, const auto &r) {
                        if (r.isAllocated())
                          return us + r.alloc_info->user_size;
                        else
                          return us;
                      }));

  // Compute memory requirements
  // This is the start of the infinite region
  // Remember to align to page
  this->stats_.max_rq =
      std::max(this->stats_.max_rq,
               INTROUNDTO(this->region_list_.crbegin()->start, 0x1000));
  // Compute shadow memory requirements
  // This is max_rq, divided by granularity, rounded up to nearest power of two
  if (this->stats_.max_rq > 0) {
    this->stats_.max_shad_rq = 1;
    while (this->stats_.max_shad_rq < this->stats_.max_rq / this->granularity_)
      this->stats_.max_shad_rq *= 2;
  } else {
    this->stats_.max_shad_rq = 0;
  }
  // Compute sum of above
  this->stats_.max_tot_rq = this->stats_.max_rq + this->stats_.max_shad_rq;

  // Compute overheads
  this->stats_.max_ov =
      (double)this->stats_.max_rq / (double)this->stats_.max_us;
  this->stats_.max_shad_ov =
      (double)this->stats_.max_shad_rq / (double)this->stats_.max_us;
  this->stats_.max_tot_ov =
      (double)this->stats_.max_tot_rq / (double)this->stats_.max_us;
}

void Simulator::cudaMalloc(uint64_t tag, size_t sz) {

  // Remember to update
  this->updateStats();
}

void Simulator::cudaFree(uint64_t tag) {

  // Remember to update
  this->updateStats();
}
