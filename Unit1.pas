unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Menus, Vcl.Buttons,
  Vcl.ExtCtrls;

type
  TSAM = class(TForm)
    CommandPrompt: TEdit;
    CommandLine: TMemo;
    MenuPanel: TPanel;
    ContextPanel: TPanel;
    TrainingPanel: TPanel;
    procedure PanelMouseEnter(Sender: TObject);
    procedure PanelMouseLeave(Sender: TObject);
    procedure TrainingPanelClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  SAM: TSAM;

implementation

{$R *.dfm}

// Menu buttons hover color event
procedure TSAM.PanelMouseEnter(Sender: TObject);
begin
  (Sender as TPanel).Color := clMenuHighlight;
end;

procedure TSAM.PanelMouseLeave(Sender: TObject);
begin
  (Sender as TPanel).Color := Self.Color;
end;

procedure TSAM.TrainingPanelClick(Sender: TObject);
begin

end;

// Menu buttons click event

end.
