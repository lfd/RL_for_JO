SELECT * FROM movie_info_idx AS miidx, title AS t WHERE miidx.movie_id = t.id AND t.id = miidx.movie_id;