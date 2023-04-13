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

  // If starts and ends are equal, then equal
  if (this->start == that.end && this->end == that.end)
    return std::partial_ordering::equivalent;

  // If this start is to the left, check if this end is also to the left,
  // otherwise incomparable.
  if (this->start <= that.start) {
    if (this->end <= that.start)
      return std::partial_ordering::less;
    return std::partial_ordering::unordered;
  }

  // Same as above, but with the roles reversed
  if (this->start >= that.start) {
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

bool Simulator::Region::isAlloc() const { return this->alloc_info.has_value(); }

bool Simulator::Region::isFree() const { return !this->alloc_info.has_value(); }

Simulator::Simulator()
    : granularity_{1ul << ARGS.get<size_t>("--scale")}, stats_{} {
  // Initialize the freelist with one huge chunk
  this->region_list_.push_back({.start = 0,
                                .end = std::numeric_limits<size_t>::max(),
                                .alloc_info = std::nullopt});
  // Update the statistics
  // We mainly do this for invariant checking
  this->updateStats();
}

Stats Simulator::getStats() const { return this->stats_; }

void Simulator::updateStats() {
  // Do assertions here
  // These take quadratic time, so only do them in debug builds
#ifndef NDEBUG

  // Debug printouts
  if (ARGS.get<bool>("--debug")) {
    std::cout << "REGION LIST:\n";
    for (const auto &r : this->region_list_)
      std::cout << "- " << (r.isFree() ? 'F' : 'A') << ' ' << r.start << ' '
                << r.end << '\n';
    std::cout << std::endl;
  }

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
  assert(!this->region_list_.crbegin()->isAlloc() &&
         "The infinite region must not be allocated");

  // Assert that all the elements of the region list are contiguous
  std::accumulate(this->region_list_.cbegin(), this->region_list_.cend(), 0ul,
                  [](size_t p, const auto &r) {
                    assert(r.start == p && "Regions must be contiguous");
                    return r.end;
                  });
  // Assert that all the elements of the region list are sorted
  for (auto ri = this->region_list_.cbegin(); ri != this->region_list_.cend();
       ri++) {
    for (auto rj = this->region_list_.cbegin(); rj != ri; rj++) {
      assert(*rj < *ri && "Region list must be sorted");
    }
  }
  // Assert that they alternate between free and allocated
  // The first one should be free
  std::accumulate(this->region_list_.cbegin(), this->region_list_.cend(), false,
                  [](bool expected_alloc, const auto &r) {
                    assert(r.isAlloc() == expected_alloc &&
                           "Regions must alternate between free and allocated");
                    return !expected_alloc;
                  });

  // All allocations must be aligned
  for (const auto &r : this->region_list_)
    if (r.isAlloc())
      assert(r.start % this->granularity_ == 0 &&
             "Allocation starts must be aligned");
  // Check redzone constraints
  for (auto r = this->region_list_.cbegin(); r != this->region_list_.cend();
       r++) {
    if (r->isAlloc()) {
      assert(std::prev(r)->size() >= r->alloc_info->redzoneSize() &&
             "Left redzone constraint violated");
      assert(std::next(r)->size() >= r->alloc_info->redzoneSize() &&
             "Right redzone constraint violated");
    }
  }
#endif

  // Sum all the user sizes
  this->stats_.max_us = std::max(
      this->stats_.max_us,
      std::accumulate(this->region_list_.cbegin(), this->region_list_.cend(),
                      0ul, [](size_t us, const auto &r) {
                        if (r.isAlloc())
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

  // The tag should not exist in the list
  assert(std::none_of(this->region_list_.cbegin(), this->region_list_.cend(),
                      [tag](const auto &r) {
                        return r.isAlloc() && r.alloc_info->tag == tag;
                      }) &&
         "Tag already in list");

  // Create an allocation info structure for the size
  const AllocationInfo ai = {.tag = tag, .user_size = sz};

  // Find a region in which we can insert this allocation
  // Keep track of the extra padding we need before
  std::list<Region>::iterator reg;
  size_t start_pad;
  for (reg = this->region_list_.begin(), start_pad = 0;
       reg != this->region_list_.end(); reg++) {

    // Only work with free regions
    if (reg->isAlloc())
      continue;

    // Most of this loop computes the required size of the region
    size_t required_size = 0;

    // Compute the redzone we need before
    // If we have a previous region, check how much redzone it needs
    size_t redzone_before = ai.redzoneSize();
    if (reg != this->region_list_.begin()) {
      assert(std::prev(reg)->isAlloc() &&
             "Region before free must be allocated");
      redzone_before =
          std::max(ai.redzoneSize(), std::prev(reg)->alloc_info->redzoneSize());
    }
    // Add to required size
    required_size += redzone_before;

    // Align
    // This is the padding we need at the start if we choose to go with this
    // region
    if ((reg->start + required_size) % this->granularity_ != 0)
      required_size += this->granularity_ -
                       ((reg->start + required_size) % this->granularity_);
    start_pad = required_size;

    // Add in what we need
    required_size += ai.user_size;

    // Compute the redzone we need after
    // If we have a next region, check how much redzone it needs
    size_t redzone_after = ai.redzoneSize();
    if (std::next(reg) != this->region_list_.end()) {
      assert(std::next(reg)->isAlloc() &&
             "Region after free must be allocated");
      redzone_after =
          std::max(ai.redzoneSize(), std::next(reg)->alloc_info->redzoneSize());
    }
    // Add to required size
    required_size += redzone_after;

    // Check whether the size requirement is met
    if (reg->size() >= required_size)
      break;
  }

  // We have to have found a region
  // Also check the padding
  assert(reg != this->region_list_.end() && "Could not find suitable region");
  assert(start_pad >= ai.redzoneSize() && "Start padding too small");
  assert((reg->start + start_pad) % this->granularity_ == 0 &&
         "Bad start alignment");

  // Insert new allocation
  // Save important places
  const size_t r_start = reg->start;
  const size_t a_start = r_start + start_pad;
  const size_t a_end = a_start + ai.user_size;
  this->region_list_.insert(
      reg, {.start = r_start, .end = a_start, .alloc_info = std::nullopt});
  this->region_list_.insert(reg,
                            {.start = a_start, .end = a_end, .alloc_info = ai});
  reg->start = a_end;
  assert(reg->end >= reg->start && "Regions must start before they end");

  // Remember to update
  this->updateStats();
}

void Simulator::cudaFree(uint64_t tag) {

  // There should only be one element matching the tag
  assert(std::count_if(this->region_list_.cbegin(), this->region_list_.cend(),
                       [tag](const auto &r) {
                         return r.isAlloc() && r.alloc_info->tag == tag;
                       }) == 1 &&
         "Not exactly one element matching tag");

  // Find the tag
  auto cur = std::find_if(
      this->region_list_.begin(), this->region_list_.end(),
      [tag](const auto &r) { return r.isAlloc() && r.alloc_info->tag == tag; });
  assert(cur != this->region_list_.end() && "No matching tag");

  // Get the previous and next regions
  assert(cur != this->region_list_.cbegin() &&
         "Allocated region at start of list");
  assert(std::next(cur) != this->region_list_.cend() &&
         "Allocated region at end of list");
  auto prev = std::prev(cur);
  auto next = std::next(cur);
  assert(prev->isFree() && "Region before allocated must be free");
  assert(next->isFree() && "Region after allocated must be free");

  // Make previous eat cur and next
  prev->end = next->end;
  this->region_list_.erase(cur);
  this->region_list_.erase(next);

  // Remember to update
  this->updateStats();
}
