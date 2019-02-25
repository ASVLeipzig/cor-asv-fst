import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .decode import FSTCorrection

@click.command()
@ocrd_cli_options
def cor_asv_fst(*args, **kwargs):
    return ocrd_cli_wrap_processor(FSTCorrection, *args, **kwargs)
