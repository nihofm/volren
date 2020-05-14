#include "query.h"

// -------------------------------------------------------
// CPU timer query

TimerQuery::TimerQuery(const std::string& name, size_t samples) : NamedMap(name), buf(samples) {}

TimerQuery::~TimerQuery() {}

void TimerQuery::start() { start_time = std::chrono::system_clock::now(); }

void TimerQuery::end() {
    buf.put(std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
                std::chrono::system_clock::now() - start_time).count());
}

float TimerQuery::get() const { return buf.avg(); }

// -------------------------------------------------------
// GPU timer query

TimerQueryGL::TimerQueryGL(const std::string& name, size_t samples) : NamedMap(name), buf(samples) {
    glGenQueries(2, query_ids[0]);
    glGenQueries(2, query_ids[1]);
    glQueryCounter(query_ids[1][0], GL_TIMESTAMP);
    glQueryCounter(query_ids[1][1], GL_TIMESTAMP);
}

TimerQueryGL::~TimerQueryGL() {
    glDeleteQueries(2, query_ids[0]);
    glDeleteQueries(2, query_ids[1]);
}

void TimerQueryGL::start() { glQueryCounter(query_ids[0][0], GL_TIMESTAMP); }

void TimerQueryGL::end() {
    glQueryCounter(query_ids[0][1], GL_TIMESTAMP);
    std::swap(query_ids[0], query_ids[1]); // switch front/back buffer
    glGetQueryObjectui64v(query_ids[0][0], GL_QUERY_RESULT, &start_time);
    glGetQueryObjectui64v(query_ids[0][1], GL_QUERY_RESULT, &stop_time);
    buf.put((stop_time - start_time) / 1000000.0);
}

float TimerQueryGL::get() const { return buf.avg(); }
