#include "Simulator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "arguments.hpp"

using namespace gasan::alloc_sim;

/**
 * \brief Rounds `v` to the nearest multiple of `m`
 *
 * Only works if both arguments are integers.
 */
#define INTROUNDTO(v, m) ((((v) + (m)-1) / (m)) * (m))

std::ostream &gasan::alloc_sim::operator<<(std::ostream &os, const Stats &s) {
  os << "max_sz " << s.max_sz << '\n';
  os << "max_rq " << s.max_rq << '\n';
  os << "max_shad_rq " << s.max_shad_rq << '\n';
  os << "max_shad_ov " << s.max_shad_ov << '\n';
  os << "max_redzone_rq " << s.max_redzone_rq << '\n';
  os << "max_redzone_ov " << s.max_redzone_ov << '\n';
  os << "max_extra_rq " << s.max_extra_rq << '\n';
  os << "max_extra_ov " << s.max_extra_ov << '\n';
  return os;
}

std::partial_ordering Simulator::Region::operator<=>(const Region &that) const {

  // Assert both regions have their ends after their starts
  assert((this->isInfinite() || *this->end >= this->start) &&
         "This region's end must be after its start");
  assert((that.isInfinite() || *that.end >= that.start) &&
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
    if (this->isInfinite())
      return std::partial_ordering::unordered;
    // If we are strictly to the left
    if (*this->end <= that.start)
      return std::partial_ordering::less;
    // We aren't strictly to the left
    return std::partial_ordering::unordered;
  }

  // Same as above, but with the roles reversed
  if (that.start < this->start) {
    if (that.isInfinite())
      return std::partial_ordering::unordered;
    if (*that.end <= this->start)
      return std::partial_ordering::greater;
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

bool Simulator::Region::isInfinite() const { return !this->end.has_value(); }
bool Simulator::Region::isFinite() const { return this->end.has_value(); }

Simulator::Simulator()
    : granularity_(1 << ARGS.get<size_t>("--scale")), stats_() {
  // Initialize the freelist with one huge chunk
  this->free_list_.emplace(0, std::nullopt);
  // Update the statistics
  // We mainly do this for invariant checking
  this->updateStats();
}

Stats Simulator::getStats() const { return this->stats_; }

void Simulator::updateStats() {
  // Do assertions here
  // These take quadratic time, so only do them in debug builds
#ifndef NDEBUG

  // Check that the greatest element of the free list is an infinite region
  // For this, it must have at least one element
  // This implies no other infinite regions with the total order check later
  assert(!this->free_list_.empty() && "Free list must not be empty");
  assert(this->free_list_.rbegin()->isInfinite() &&
         "Last region in free list must be infinite");

  // Check that all regions have their ends after their starts
  for (const auto &r : this->free_list_)
    assert((r.isInfinite() || *r.end >= r.start) &&
           "Free region end must be after start");
  for (const auto &[ak, av] : this->alloc_list_)
    assert((av.region.isInfinite() || *av.region.end >= av.region.start) &&
           "Allocated region end must be after start");

  // Check that all regions are aligned to granularity
  for (const auto &r : this->free_list_) {
    assert(r.start % this->granularity_ == 0 &&
           "Free region start must be aligned");
    assert(r.isInfinite() || *r.end % this->granularity_ == 0 &&
                                 "Free region end must be aligned");
  }
  for (const auto &[ak, av] : this->alloc_list_) {
    assert(av.region.start % this->granularity_ == 0 &&
           "Allocated region start must be aligned");
    assert(av.region.isInfinite() ||
           *av.region.end % this->granularity_ == 0 &&
               "Allocated region end must be aligned");
  }

  // Check that all allocations have a required size that's at least the size
  // needed by the user. Also check that the required size is aligned.
  for (const auto &[ak, av] : this->alloc_list_) {
    assert(av.required_size >= av.size && "Requested size too small");
    assert(av.required_size % this->granularity_ == 0 &&
           "Requested size not aligned");
  }

  // Check that no allocation is backed by an infinite region
  for (const auto &[ak, av] : this->alloc_list_)
    assert(av.region.isFinite() &&
           "Allocations must not be backed by infinite regions");

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
  // list. Also check that they are totally ordered
  for (const auto &f : this->free_list_)
    for (const auto &[ak, av] : this->alloc_list_)
      assert((f < av.region || f > av.region) &&
             "Allocation and Free lists not disjoint");
#endif

  this->stats_.max_sz = std::max(
      this->stats_.max_sz,
      std::accumulate(
          this->alloc_list_.begin(), this->alloc_list_.end(), 0ul,
          [](const auto &l, const auto &r) { return l + r.second.size; }));
  this->stats_.max_rq =
      std::max(this->stats_.max_rq, this->free_list_.rbegin()->start);

  this->stats_.max_shad_rq =
      INTROUNDTO(this->stats_.max_rq, this->granularity_) / this->granularity_;
  this->stats_.max_shad_ov =
      (double)this->stats_.max_shad_rq / (double)this->stats_.max_sz;

  this->stats_.max_redzone_rq = this->stats_.max_rq - this->stats_.max_sz;
  this->stats_.max_redzone_ov =
      (double)this->stats_.max_redzone_rq / (double)this->stats_.max_sz;

  this->stats_.max_extra_rq =
      this->stats_.max_shad_rq + this->stats_.max_redzone_rq;
  this->stats_.max_extra_ov =
      (double)this->stats_.max_extra_rq / (double)this->stats_.max_sz;
}

void Simulator::cudaMalloc(uint64_t tag, size_t sz) {

  // Compute the required size with the redzones
  size_t req_sz = sz;
  {
    // Base is just the normal base rounded up to the granularity
    size_t rz_base =
        INTROUNDTO(ARGS.get<size_t>("--redzone-base"), this->granularity_);
    // Scale is the same, but it depends on the requested size
    size_t rz_scale = std::ceil(sz * ARGS.get<double>("--redzone-scale"));
    rz_scale = INTROUNDTO(rz_scale, this->granularity_);
    // Actual redzone size is the max of those
    // And the requested size must be made a multiple of the granularity
    req_sz = req_sz + std::max(rz_base, rz_scale);
    req_sz = INTROUNDTO(req_sz, this->granularity_);
    assert(req_sz >= sz && "Requested size not large enough");
    assert(req_sz >= rz_base && "Requested size not large enough");
    assert(req_sz >= rz_scale && "Requested size not large enough");
    assert(req_sz % this->granularity_ == 0 && "Requested size not aligned");
  }

  // Find a matching region for the size
  // We should always succeed
  auto reg = std::find_if(this->free_list_.begin(), this->free_list_.end(),
                          [req_sz](const auto &r) {
                            return r.isInfinite() || *r.size() >= req_sz;
                          });
  assert(reg != this->free_list_.end() && "Failed to find a region");
  assert(reg->start % this->granularity_ == 0 && "Region must be aligned");

  // If we don't need to split
  if (reg->isFinite() && *reg->size() - req_sz <= this->granularity_) {
    // Insert the allocation
    {
      auto r = this->alloc_list_.emplace(tag, AllocationInfo{*reg, sz, req_sz});
      assert(r.second && "Multiple allocations with same tag");
    }
    // Mark the original region as not free
    this->free_list_.erase(reg);

  } else {
    // Insert the allocation
    {
      auto r = this->alloc_list_.emplace(
          tag,
          AllocationInfo{Region{reg->start, reg->start + req_sz}, sz, req_sz});
      assert(r.second && "Multiple allocations with same tag");
    }
    // Increase the start of the region in the free list
    {
      auto new_reg = *reg;
      new_reg.start += req_sz;
      this->free_list_.erase(reg);
      this->free_list_.insert(new_reg);
    }
  }

  // Remember to update
  this->updateStats();
}

void Simulator::cudaFree(uint64_t tag) {

  // Lookup the element to free
  // It will be present
  auto alloc = this->alloc_list_.find(tag);
  assert(alloc != this->alloc_list_.end() && "Could not find element to free");

  // Delete the allocation, keeping only the region
  auto c = alloc->second.region;
  this->alloc_list_.erase(alloc);
  assert(c.isFinite() && "Allocated regions must be finite");

  // Merge right if possible
  auto r = this->free_list_.upper_bound(c);
  if (r != this->free_list_.end() && *c.end == r->start) {
    c.end = r->end;
    this->free_list_.erase(r);
  }

  // Merge left if possible
  // Note that the free list may be empty at this point. If it is, don't do
  // anything
  if (!this->free_list_.empty()) {
    auto l = this->free_list_.lower_bound(c);
    // Check that the lower bound we got isn't at the start of the list. If it
    // is, then there are no elements to merge.
    if (l != this->free_list_.begin()) {
      l--;
      if (l->end == c.start) {
        c.start = l->start;
        this->free_list_.erase(l);
      }
    }
  }

  // Insert the new region
  this->free_list_.insert(c);

  // Remember to update
  this->updateStats();
}
