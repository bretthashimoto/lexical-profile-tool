1. Investigate weird words that aren't being tagged correctly
2. Filter COCA lemmas list to get rid of garbage
3. Add lemmatization to NGSL, NAWL, and AVL lists. (AVL and NAWL done --
   NAWL is now POS-tagged too, using NAWL_1.2 data.csv.) Want to make
   similar fixes (lemmatization + POS data where possible) to all the
   pre-loaded word lists -- NGSL is the one still missing POS data.
4. Investigate words on the word lists that don't seem to be working correctly
5. Combine the exclude numerals and proper nouns settings ection to one place
6. Word families? not just lemmas?
7. get help testing out features
8. Add a tutorial to the interface
9. clean up and get rid of dead code and other unnecessary stuff
10. Brainstorm other things to improve about the app
11. Have Claude help brainstorm new features
12. Find default frequency reference data for Spanish, French, and German,
    like NGSL/COCA for English -- maybe SUBTLEX for all four languages
    (SUBTLEX-US, SUBTLEX-ESP, SUBTLEX-FR, SUBTLEX-DE)? Should be pretty
    easy in theory.
13. Once deployed on BYU's server, run scripts/benchmark_n_process.py
    there and recalibrate auto_n_process's default threshold (currently
    6,000 texts, tuned only for this Windows dev machine) based on the
    real numbers -- check its start-method (fork vs spawn) and cgroup
    CPU quota output especially, since shared hosting may cap cores well
    below what os.cpu_count() reports.
