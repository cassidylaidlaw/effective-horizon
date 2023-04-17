
#include <boost/math/distributions/binomial.hpp>

extern "C" {
    double binomcdf(long n, double p, long k) {
        return boost::math::cdf(boost::math::binomial(n, p), k);
    }
    long binominvcdf(long n, double p, double q) {
        return boost::math::quantile(boost::math::binomial(n, p), q);
    }
}
