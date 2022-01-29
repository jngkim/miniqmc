//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/** @file NewTimer.cpp
 * @brief Implements timer_manager
 */
#include "TimerManager.h"
#include <limits>
#include <cstdio>
#include <algorithm>
#include <array>
#include <stdexcept>
#include "Configuration.h"

namespace qmcplusplus
{
TimerManager<NewTimer> timer_manager;

const std::array<std::string, num_timer_levels> timer_level_names = {"none", "coarse", "medium", "fine"};

const char TIMER_STACK_SEPARATOR = '/';

template<class TIMER>
void TimerManager<TIMER>::initializeTimer(TIMER& t)
{
  if (t.get_name().find(TIMER_STACK_SEPARATOR) != std::string::npos)
    app_log() << "Warning: Timer name (" << t.get_name() << ") should not contain the character "
              << TIMER_STACK_SEPARATOR << std::endl;

  if (timer_name_to_id.find(t.get_name()) == timer_name_to_id.end())
  {
    t.set_id(max_timer_id);
    timer_id_name[t.get_id()]      = t.get_name();
    timer_name_to_id[t.get_name()] = t.get_id();
    if (max_timer_id >= std::numeric_limits<timer_id_t>::max())
    {
      max_timers_exceeded = true;
      app_log() << "Number of timers exceeds limit (" << static_cast<int>(std::numeric_limits<timer_id_t>::max())
                << ").   Adjust timer_id_t in NewTimer.h and recompile." << std::endl;
    }
    else
      max_timer_id++;
  }
  else
    t.set_id(timer_name_to_id[t.get_name()]);

  t.set_active_by_timer_threshold(timer_threshold);
}

template<class TIMER>
TIMER* TimerManager<TIMER>::createTimer(const std::string& myname, timer_levels mytimer)
{
  TIMER* t = nullptr;
  {
    const std::lock_guard<std::mutex> lock(timer_list_lock_);
    timer_list_.push_back(std::make_unique<TIMER>(myname, this, mytimer));
    t = timer_list_.back().get();
    initializeTimer(*t);
  }
  return t;
}

template<class TIMER>
void TimerManager<TIMER>::push_timer(TIMER* t)
{
  // current_timer() can be nullptr when the stack was empty.
  if (t == current_timer())
  {
    std::cerr << "Timer " << t->get_name()
              << " instance is already at the top of the stack. "
                 "start() is being called again. This often happens when stop() is not paired properly with start(). "
              << "ScopedTimer uses RAII and manages timer start/stop more safely." << std::endl;
    throw std::runtime_error("TimerManager push_timer error!");
  }
  else
    CurrentTimerStack.push_back(t);
}

template<class TIMER>
void TimerManager<TIMER>::pop_timer(TIMER* t)
{
  TIMER* stack_top = current_timer();
  if (stack_top == nullptr)
  {
    std::cerr << "Timer stack pop failed on an empty stack! Requested \"" << t->get_name() << "\"." << std::endl;
    throw std::runtime_error("TimerManager pop_timer error!");
  }
  else if (t != stack_top)
  {
    std::cerr << "Timer stack pop not matching push! "
              << "Expecting \"" << t->get_name() << "\" but \"" << stack_top->get_name() << "\" is on the top."
              << std::endl;
    throw std::runtime_error("TimerManager pop_timer error!");
  }
  else
    CurrentTimerStack.pop_back();
}

template<class TIMER>
void TimerManager<TIMER>::reset()
{
  for (int i = 0; i < timer_list_.size(); i++)
    timer_list_[i]->reset();
}

template<class TIMER>
void TimerManager<TIMER>::set_timer_threshold(const timer_levels threshold)
{
  timer_threshold = threshold;
  for (int i = 0; i < timer_list_.size(); i++)
    timer_list_[i]->set_active_by_timer_threshold(timer_threshold);
}

template<class TIMER>
void TimerManager<TIMER>::set_timer_threshold(const std::string& threshold)
{
  const auto it = std::find(timer_level_names.begin(), timer_level_names.end(), threshold);
  if (it != timer_level_names.end())
    set_timer_threshold(static_cast<timer_levels>(std::distance(timer_level_names.begin(), it)));
  else
  {
    std::cerr << "Unknown timer level: " << threshold << " , current level: " << timer_level_names[timer_threshold]
              << std::endl;
  }
}

template<class TIMER>
std::string TimerManager<TIMER>::get_timer_threshold_string() const
{
  return timer_level_names[timer_threshold];
}


template<class TIMER>
void TimerManager<TIMER>::collate_flat_profile(FlatProfileData& p)
{
  for (int i = 0; i < timer_list_.size(); ++i)
  {
    TIMER& timer = *timer_list_[i];
    nameList_t::iterator it(p.nameList.find(timer.get_name()));
    if (it == p.nameList.end())
    {
      int ind                      = p.nameList.size();
      p.nameList[timer.get_name()] = ind;
      p.timeList.push_back(timer.get_total());
      p.callList.push_back(timer.get_num_calls());
    }
    else
    {
      int ind = (*it).second;
      p.timeList[ind] += timer.get_total();
      p.callList[ind] += timer.get_num_calls();
    }
  }
}

struct ProfileData
{
  double time;
  double calls;

  ProfileData& operator+=(const ProfileData& pd)
  {
    time += pd.time;
    calls += pd.calls;
    return *this;
  }
};

int get_level(const std::string& stack_name)
{
  int level = 0;
  for (int i = 0; i < stack_name.length(); i++)
    if (stack_name[i] == TIMER_STACK_SEPARATOR)
      level++;

  return level;
}

std::string get_leaf_name(const std::string& stack_name)
{
  int pos = stack_name.find_last_of(TIMER_STACK_SEPARATOR);
  if (pos == std::string::npos)
    return stack_name;

  return stack_name.substr(pos + 1, stack_name.length() - pos);
}

template<class TIMER>
void TimerManager<TIMER>::get_stack_name_from_id(const StackKey& key, std::string& stack_name)
{
  for (int i = 0; i < StackKey::max_level; i++)
  {
    std::string& timer_name = timer_id_name[key.get_id(i)];
    if (key.get_id(i) == 0)
      break;
    if (i > 0)
      stack_name += TIMER_STACK_SEPARATOR;

    stack_name += timer_name;
  }
}

template<class TIMER>
void TimerManager<TIMER>::collate_stack_profile(StackProfileData& p)
{
#ifdef USE_STACK_TIMERS
  // Put stacks from all timers into one data structure
  // By naming the timer stacks as 'timer1/timer2', etc, the ordering done by the
  // map's keys will also place the stacks in depth-first order.
  // The order in which sibling timers are encountered in the code is not
  // preserved. They will be ordered alphabetically instead.
  std::map<std::string, ProfileData> all_stacks;
  for (int i = 0; i < timer_list_.size(); ++i)
  {
    TIMER& timer                                     = *timer_list_[i];
    std::map<StackKey, double>::iterator stack_id_it = timer.get_per_stack_total_time().begin();
    for (; stack_id_it != timer.get_per_stack_total_time().end(); stack_id_it++)
    {
      ProfileData pd;
      const StackKey& key = stack_id_it->first;
      std::string stack_name;
      get_stack_name_from_id(key, stack_name);
      pd.time  = timer.get_total(key);
      pd.calls = timer.get_num_calls(key);

      all_stacks[stack_name] += pd;
    }
  }

  // Fill in the output data structure (but don't compute exclusive time yet)
  std::map<std::string, ProfileData>::iterator si = all_stacks.begin();
  int idx                                         = 0;
  for (; si != all_stacks.end(); ++si)
  {
    std::string stack_name = si->first;
    p.nameList[stack_name] = idx;
    p.names.push_back(stack_name);
    p.timeList.push_back(si->second.time);
    p.timeExclList.push_back(si->second.time);
    p.callList.push_back(si->second.calls);
    idx++;
  }

  // Subtract times of immediate children to get exclusive time
  for (idx = 0; idx < p.timeList.size(); idx++)
  {
    int start_level = get_level(p.names[idx]);
    for (int i = idx + 1; i < p.timeList.size(); i++)
    {
      int level = get_level(p.names[i]);
      if (level == start_level + 1)
        p.timeExclList[idx] -= p.timeExclList[i];

      if (level == start_level)
        break;
    }
  }
#endif
}

template<class TIMER>
void TimerManager<TIMER>::print(std::ostream& out)
{
  if (timer_threshold <= timer_level_none)
    return;
#ifdef ENABLE_TIMERS
#ifdef USE_STACK_TIMERS
  out << "Stack timer profile" << std::endl;
  print_stack(out);
#else
  out << "Flat profile" << std::endl;
  print_flat(out);
#endif
#endif
}

template<class TIMER>
void TimerManager<TIMER>::print_flat(std::ostream& out)
{
#ifdef ENABLE_TIMERS
  FlatProfileData p;

  collate_flat_profile(p);

#pragma omp master
  {
    const int bufsize = 256;
    char tmpout[bufsize];
    std::map<std::string, int>::iterator it(p.nameList.begin()), it_end(p.nameList.end());
    while (it != it_end)
    {
      int i = (*it).second;
      //if(callList[i]) //skip zeros
      snprintf(tmpout, bufsize, "%-40s  %9.4f  %13ld  %16.9f  %12.6f TIMER\n", (*it).first.c_str(), p.timeList[i],
               p.callList[i],
               p.timeList[i] / (static_cast<double>(p.callList[i]) + std::numeric_limits<double>::epsilon()),
               p.timeList[i] / static_cast<double>(omp_get_max_threads()));
      out << tmpout;
      ++it;
    }
  }
#endif
}


void pad_string(const std::string& in, std::string& out, int field_len)
{
  int len     = in.size();
  int pad_len = std::max(field_len - len, 0);
  std::string pad_str(pad_len, ' ');
  out = in + pad_str;
}

template<class TIMER>
void TimerManager<TIMER>::print_stack(std::ostream& out)
{
#ifdef ENABLE_TIMERS
  StackProfileData p;

  collate_stack_profile(p);

  if (timer_max_level_exceeded)
  {
    app_warning() << "Maximum stack level (" << StackKey::max_level << ") exceeded.  Results may be incorrect."
                  << std::endl;
    app_warning() << "Adjust StackKey in NewTimer.h and recompile." << std::endl;
  }

  int indent_len   = 2;
  int max_name_len = 0;
  for (int i = 0; i < p.names.size(); i++)
  {
    std::string stack_name = p.names[i];
    int level              = get_level(stack_name);
    std::string name       = get_leaf_name(stack_name);
    int name_len           = name.size() + indent_len * level;
    max_name_len           = std::max(name_len, max_name_len);
  }

  const int bufsize = 256;
  char tmpout[bufsize];
  std::string timer_name;
  pad_string("Timer", timer_name, max_name_len);

  snprintf(tmpout, bufsize, "%s  %-9s  %-9s  %-10s  %-13s\n", timer_name.c_str(), "Inclusive_time", "Exclusive_time",
           "Calls", "Time_per_call");
  out << tmpout;

  for (int i = 0; i < p.names.size(); i++)
  {
    std::string stack_name = p.names[i];
    int level              = get_level(stack_name);
    std::string name       = get_leaf_name(stack_name);
    std::string indent_str(indent_len * level, ' ');
    std::string indented_str = indent_str + name;
    std::string padded_name_str;
    pad_string(indented_str, padded_name_str, max_name_len);
    snprintf(tmpout, bufsize, "%s  %9.4f  %9.4f  %13ld  %16.9f\n", padded_name_str.c_str(), p.timeList[i],
             p.timeExclList[i], p.callList[i],
             p.timeList[i] / (static_cast<double>(p.callList[i]) + std::numeric_limits<double>::epsilon()));
    out << tmpout;
  }
#endif
}

template<class TIMER>
XMLNode* TimerManager<TIMER>::TimerManager::output_timing(XMLDocument& doc)
{
  XMLNode* timing_root = doc.NewElement("timing");
#ifdef ENABLE_TIMERS
  StackProfileData p;

  collate_stack_profile(p);

  timing_root->InsertEndChild(
      MakeTextElement(doc, "max_stack_level_exceeded", timer_max_level_exceeded ? "yes" : "no"));
  timing_root->InsertEndChild(MakeTextElement(doc, "max_timers_exceeded", max_timers_exceeded ? "yes" : "no"));

  std::vector<XMLNode*> node_stack;
  node_stack.push_back(timing_root);
  XMLNode* current_root = timing_root;
  for (int i = 0; i < p.names.size(); i++)
  {
    std::string stack_name = p.names[i];
    int level              = get_level(stack_name);
    std::string name       = get_leaf_name(stack_name);

    std::string indent_str(2 * level, ' ');

    XMLNode* timer = doc.NewElement("timer");
    current_root->InsertEndChild(timer);

    timer->InsertEndChild(MakeTextElement(doc, "name", name));
    timer->InsertEndChild(MakeTextElement(doc, "time_incl", std::to_string(p.timeList[i])));
    timer->InsertEndChild(MakeTextElement(doc, "time_excl", std::to_string(p.timeExclList[i])));
    timer->InsertEndChild(MakeTextElement(doc, "calls", std::to_string(p.callList[i])));

    int next_level = level;
    if (i + 1 < p.names.size())
    {
      next_level = get_level(p.names[i + 1]);
    }

    if (next_level > level)
    {
      //XMLNode* next_node = doc.addChild(timer, "includes");
      XMLNode* next_node = doc.NewElement("includes");
      timer->InsertEndChild(next_node);
      node_stack.push_back(next_node);
      current_root = next_node;
    }
    if (next_level < level)
    {
      for (int j = 0; j < level - next_level; j++)
      {
        node_stack.pop_back();
        current_root = node_stack.back();
      }
    }
  }
#endif
  return timing_root;
}

template class TimerManager<NewTimer>;
} // namespace qmcplusplus