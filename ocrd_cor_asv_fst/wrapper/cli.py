import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .decode import PageXMLProcessor

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_fst(*args, **kwargs):
    return ocrd_cli_wrap_processor(PageXMLProcessor, *args, **kwargs)
