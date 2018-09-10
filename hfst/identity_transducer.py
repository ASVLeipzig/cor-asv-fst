import hfst
import error_transducer as et



identity_transducer = hfst.regex('?*')
identity_transducer.remove_epsilons()
identity_transducer.minimize()

#et.save_transducer('identity_transducer.hfst', identity_transducer)


morphology_transducer = et.load_transducer('wwm/rules.fsm')

morphology_transducer.disjunct(identity_transducer)

et.save_transducer('morphology_with_identity.hfst', morphology_transducer)
