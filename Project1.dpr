program Project1;

uses
  Vcl.Forms,
  Unit1 in 'Unit1.pas' {SAM},
  Vcl.Themes,
  Vcl.Styles;

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TSAM, SAM);
  Application.Run;
end.
