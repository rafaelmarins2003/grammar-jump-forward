# Começar a trabalhar no C++

O experimento anterior terminou segundo seus critérios. O protótipo, os dados e os
diagnósticos continuam úteis para estudar, melhorar a implementação e escrever um relato
pessoal. Os relatórios originais registram o resultado daquela versão.

## Código disponível

O worktree original está em `/home/rafaelmarins2003/scripts/llama.cpp-jf`, na branch
`grammar-jump-forward`, sobre o commit `95ef7fc16054e63b427a3ef00188e055ef7586d8`.
O patch histórico já está aplicado. O vínculo Git com `~/scripts/llama.cpp` foi reparado
depois da mudança de pasta. Os scripts Python usam o campo `worktree` de
[`config/llama_cpp.json`](../config/llama_cpp.json) para localizar o motor.

Os fontes, as bibliotecas Rust corrigidas e os binários dos experimentos anteriores estão
preservados. As configurações CMake de Debug e Release foram regeneradas para o caminho novo,
sem recompilar esses binários. Não reaplique o patch C++ nesse worktree: ele já está presente
nas alterações de trabalho.

A cópia duplicada restaurada por engano na home foi arquivada em
`.cache/source-recovery/llama.cpp-jf-2026-10-09`, neste projeto. Ela não é usada pelos scripts.
Os metadados antigos dos builds ficaram em `.cache/path-migration-2026-10-09`.

```bash
cd /home/rafaelmarins2003/scripts/llama.cpp-jf
git status --short
git diff -- common/speculative.cpp common/llguidance.cpp common/sampling.cpp
```

O diff do worktree corresponde ao exportado no primeiro experimento. A comparação foi feita
com o mesmo formato de prefixos e abreviação dos objetos Git usado no arquivo preservado.
O worktree permanece sem novos commits para você poder inspecionar cada alteração.

## Ordem de leitura

| Arquivo no llama.cpp-jf | O que procurar |
|---|---|
| `common/speculative.cpp` | `common_speculative_impl_grammar`: solicita os tokens e limita o draft |
| `common/sampling.cpp` | `common_sampler_ff_tokens`: verifica se a gramática está ativa e se o caminho é suportado |
| `common/llguidance.cpp` | `llama_sampler_llg_ff_tokens`: chama a API, valida IDs e limita tokens de fim |
| `tools/server/server-context.cpp` | Liga o sampler do slot ao proposer e usa a verificação existente |
| `tests/test-grammar-llguidance.cpp` | Limite de buffer, clonagem, reset e equivalência com máscaras de um token |

O proposer recebe o sampler depois de o último token escolhido ter sido aceito. Aceitar esse
token novamente avançaria a gramática duas vezes. Os tokens propostos ainda passam pelo
modelo e pela verificação real, preenchendo o cache KV. A gramática restringe a escolha;
ela não calcula os estados internos do modelo.

O controle Python que executa os servidores está em `src/jumpforward/parity.py`.
O experimento principal está em `src/jumpforward/prototype.py` e o diagnóstico em
`src/jumpforward/replication.py`. Os arquivos numerados em `scripts/` são suas entradas.

## Primeiro build para estudar

Comece pelo Debug em CPU, que permite testar o sampler e usar um depurador sem ocupar a GPU.
A configuração já está atualizada; use os comandos abaixo para recompilar após suas mudanças.
Rode-os da raiz deste projeto para carregar o `mise.toml`.

```bash
mise exec -- cmake -S /home/rafaelmarins2003/scripts/llama.cpp-jf \
  -B /home/rafaelmarins2003/scripts/llama.cpp-jf/build-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF -DLLAMA_LLGUIDANCE=ON \
  -DLLAMA_OPENSSL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

mise exec -- cmake --build /home/rafaelmarins2003/scripts/llama.cpp-jf/build-debug \
  --target llama-server test-grammar-llguidance test-sampling -j 4
ctest --test-dir /home/rafaelmarins2003/scripts/llama.cpp-jf/build-debug \
  -R '^(test-grammar-llguidance|test-sampling)$' --output-on-failure
```

O patch de llguidance corrige uma cópia que ultrapassava a capacidade fornecida ao buffer.
É um requisito concreto deste protótipo com a dependência fixada. Ela já está aplicada nos
builds Debug e Release deste worktree: não repita `git apply`. Se criar um build em outra
pasta, siga o procedimento para dependência nova em
[prototype-reproduction.md](prototype-reproduction.md) antes de executar o proposer.

O build Release e os protocolos estão em [environment.md](environment.md),
[prototype-reproduction.md](prototype-reproduction.md) e
[replication-reproduction.md](replication-reproduction.md). O teste `smoke` do script 08
também espera uma cópia do runtime sem o patch em `build-pre-grammar/bin`; essa cópia está
preservada no worktree original. Uma recompilação pode mudar os hashes:
prepare um novo run em vez de tentar continuar os runs históricos selados.

## Primeiras mudanças que fazem sentido

1. Entender e reproduzir o caminho completo com os testes existentes.
2. Medir onde o tempo é gasto: consulta de tokens forçados, alocações, amostragem,
   verificação e chamadas ao modelo. O helper atualmente cria dois buffers por consulta;
   isso é um ponto para investigar, não uma economia já demonstrada.
3. Fazer uma alteração pequena e medir contra a mesma versão com a estratégia desligada.
   Mantenha os critérios, prompts e falhas registrados antes de comparar os resultados.

Os diagnósticos anteriores mostraram que mudanças de conteúdo desapareceram no controle
com lotes físicos de um token. Isso sustenta sensibilidade ao formato de execução, mas
não estabelece a diferença numérica exata. Investigar os logits e avaliar o acerto das
respostas são caminhos mais informativos que apenas exigir texto idêntico.

Use os casos já observados para desenvolvimento. Uma confirmação de uma versão otimizada
precisa de uma seleção nova ou de declarar explicitamente que reutiliza dados conhecidos.
Você pode testar outras configurações sem transformar uma tentativa favorável em prova
de melhoria geral.

## Blog pessoal e eventual contribuição upstream

Um artigo já pode mostrar o problema, a integração, os resultados nos dois modelos,
as diferenças de resposta e o que os controles permitiram concluir. Use os números dos
[resultados iniciais](../runs/2026-09-12-prototype/cold-report.md) e da
[replicação](../runs/2026-09-12-replication/report.md), deixando claro o baseline, modelo,
hardware e alcance da avaliação. Jump-forward e a detecção de tokens forçados têm trabalhos
anteriores; a contribuição deste protótipo é a integração e sua avaliação no llama.cpp.

Uma sequência possível para o texto: problema e hipótese; implementação; metodologia;
resultados; divergências; limitações; próximas melhorias. É possível publicar um experimento
útil que não alcançou seu próprio critério de aprovação.

Para um PR, primeiro consulte os [issues](https://github.com/ggml-org/llama.cpp/issues),
[PRs](https://github.com/ggml-org/llama.cpp/pulls) e o
[guia atual de contribuição](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md).
O upstream pede discussão prévia para funcionalidades novas, revisão humana, testes,
clareza sobre o uso de IA e capacidade de explicar e manter o código submetido.
Você deve escrever sua própria discussão e apresentação da contribuição.

O worktree usa a base histórica. Uma contribuição precisa ser portada e testada
na base atual em outra branch. As interfaces e a dependência podem ter mudado; verifique
se a mesma funcionalidade ou correção já existe. O ganho é uma parte da justificativa;
correção, compatibilidade e manutenção também serão avaliadas pelos mantenedores.
