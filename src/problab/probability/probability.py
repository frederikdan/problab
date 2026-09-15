#P(event, method="auto")
#P(event, method="exact")
#P(event, method="monte_carlo")
import numpy as np

from problab._events import _Event
from problab.probability._config import DEFAULT_PROB_NUM_SAMPLES
from problab.probability.results import ProbabilityResult
from problab.random_variables._context import _RealizationContext
from problab.validation._common import _validate_num_samples, _validate_rng, _validate_validate
from problab.validation._decorator import _validate_parameters
from problab.validation.probability._probability import _validate_event, _validate_given


@_validate_parameters(
    event=_validate_event,
    given=_validate_given,
    num_samples=_validate_num_samples,
    rng=_validate_rng,
    validate=_validate_validate,
)
def P(event: _Event,
      given: _Event | None = None,
      num_samples: int = DEFAULT_PROB_NUM_SAMPLES,
      rng: np.random.Generator | None = None,
      validate: bool = False,
      ) -> ProbabilityResult:

    num_samples = int(num_samples)

    if given is None:
        event_values = _RealizationContext(
            root_node=event._node,
            num_samples=num_samples,
            rng=rng,
            validate=validate,
        ).evaluate(event._node)

        return ProbabilityResult(
            value=float(np.mean(event_values)),
            num_successes=int(np.sum(event_values)),
            num_unconditioned_samples=num_samples
        )

    joint_event = event & given

    context = _RealizationContext(
        root_node=joint_event._node,
        num_samples=num_samples,
        rng=rng,
        validate=validate,
    )

    given_values = context.evaluate(given._node)
    num_conditioned_samples = int(np.count_nonzero(given_values))

    if num_conditioned_samples == 0:
        return ProbabilityResult(
            value=np.nan,
            num_successes=0,
            num_unconditioned_samples=num_samples,
            num_conditioned_samples=0,
        )

    joint_values = context.evaluate(joint_event._node)
    num_successes = int(np.count_nonzero(joint_values))

    return ProbabilityResult(
        value=num_successes / num_conditioned_samples,
        num_successes=num_successes,
        num_unconditioned_samples=num_samples,
        num_conditioned_samples=num_conditioned_samples,
    )



