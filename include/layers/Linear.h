/**
 * @file
 * @brief definicija modula Linear
 * @author Jakov Novak
 */

#include "Module.h"
#include <CL/cl.h>

/**
 * @class Linear
 * @brief klasa koja implementira afine transformacije, za proizvoljne dimenzije vektora: \f$ A \vec{x} + \vec{b} \f$
 */
class Linear : public Module {
  private:
    /**
     * @brief matrica s kojom mnozimo ulaz, \f$ A \f$
     */
    std::shared_ptr<autograd::Variable<Matrix>> parameters;
    /**
     * @brief vektor s kojim zbrajamo rezultat umnoška, \f$ \vec{b} \f$
     */
    std::shared_ptr<autograd::Variable<Matrix>> biases;
    /**
     * @brief dimenzije matrice \f$ A \f$
     */
    const size_t in_features, out_features;
    
  public:
    /**
     * @brief konstruktor kojemu zadajemo dimenzije matrice te zastavicu za generiranje vektora \f$ \vec{b} \f$
     *
     * matrica \f$ A \f$ i vektor \f$ \vec{b} \f$ se na početku ispune nasumičnim brojevima iz distribucije:
     * @f$ \mathcal{U}(- \sqrt{k}, + \sqrt{k}), k = 1 / \mathrm{in\_features} @f$
     *
     * @param in_features dimnezije ulaznog vektora, \f$ \vec{x} \f$
     * @param out_features dimenzije vektora izlaza 
     * @param bias bool zastavica koju postavljamo na true ako želimo koristiti vektor \f$ \vec{b} \f$
     */
    Linear(size_t in_features, size_t out_features, bool bias = false);
    
    virtual std::shared_ptr<autograd::Expression<Matrix>> forward(std::shared_ptr<autograd::Expression<Matrix>> ulaz) override final;
};
