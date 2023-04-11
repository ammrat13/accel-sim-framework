#include "Simulator.hpp"

#include <cassert>

using namespace gasan::alloc_sim;

std::ostream &gasan::alloc_sim::operator<<(std::ostream &os, const Stats &s) {
  return os;
}

std::partial_ordering Simulator::Region::operator<=>(const Region &that) const {

  // Assert both regions have their ends after their starts
  assert((!this->end || *this->end >= this->start) &&
         "This region's end must be after its start");
  assert((!that.end || *that.end >= that.start) &&
         "That region's end must be after its start");

  // If the starts are equal, return equal if the ends are, otherwise
  // incomparable
  if (this->start == that.start)
    return this->end == that.end ? std::partial_ordering::equivalent
                                 : std::partial_ordering::unordered;

  // If this start is to the left, check if that's start is also to the left,
  // otherwise incomparable. Remember to deal with infinities.
  if (this->start < that.start) {
    // If we stretch out to infinity, that can't possibly be to our right
    if (!this->end)
      return std::partial_ordering::unordered;
    // If we are strictly to the left
    if (*this->end <= that.start)
      return std::partial_ordering::less;
    // We aren't strictly to the left
    return std::partial_ordering::unordered;
  }

  // Same as above, but with the roles reversed
  if (that.start < this->start) {
    if (!that.end)
      return std::partial_ordering::unordered;
    if (*that.end <= this->start)
      return std::partial_ordering::less;
    return std::partial_ordering::unordered;
  }

  // All cases have been handled
  assert(false && "Starts not totally ordered");
}

std::optional<size_t> Simulator::Region::size() const {
  // If we stretch out to infinity, say so
  if (!this->end)
    return std::nullopt;
  // Check that the end is after the start, then return
  assert(*this->end >= this->start && "Region end must be after start");
  return *this->end - this->start;
}

Simulator::Simulator() {
  // Initialize the freelist with one huge chunk
  this->free_list_.emplace(0, std::nullopt);
}

Stats Simulator::getStats() const { return this->stats_; }

void Simulator::updateStats() {
  // Do assertions here
  // These take quadratic time, so only do them in debug builds
#ifndef NDEBUG
  // Assert that all regions have their ends after their starts
  for (const auto &r : this->free_list_)
    assert((!r.end || *r.end >= r.start) &&
           "Free region end must be after start");
  for (const auto &[ak, av] : this->alloc_list_)
    assert((!av.region.end || *av.region.end >= av.region.start) &&
           "Allocated region end must be after start");
  // Check that the free list is totally ordered
  for (const auto &a : this->free_list_)
    for (const auto &b : this->free_list_)
      assert((a <=> b) != std::partial_ordering::unordered &&
             "Free list is not totally ordered");
  // Check that the allocation list is totally ordered
  for (const auto &[ak, av] : this->alloc_list_)
    for (const auto &[bk, bv] : this->alloc_list_)
      assert((av.region <=> bv.region) != std::partial_ordering::unordered &&
             "Allocation list is not totally ordered");
  // Check that there is no overlap between the freelist and the allocation
  // list. Also checks that they are totally ordered
  for (const auto &f : this->free_list_)
    for (const auto &[ak, av] : this->alloc_list_)
      assert((f < av.region || f > av.region) &&
             "Allocation and Free lists not disjoint");
#endif
}

void Simulator::cudaMalloc(uint64_t tag, size_t sz) {
  // Remember to update
  this->updateStats();
}

void Simulator::cudaFree(uint64_t tag) {
  // Remember to update
  this->updateStats();
}
